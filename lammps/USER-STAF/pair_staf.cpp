/* -*- c++ -*- ----------------------------------------------------------
   USER-STAF — pair_style staf (Allegro-style DD via LAMMPS neigh + reverse_comm)
------------------------------------------------------------------------- */

#include "pair_staf.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <utility>

#include "staf.h"

using namespace LAMMPS_NS;
using namespace NeighConst;

PairSTAF::PairSTAF(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  /* Ghost forces from many-body chain rule → reverse_comm into owned atoms. */
  comm_reverse = 3;

  model_dir = nullptr;
  staf = nullptr;
  cut_radial = cut_angular = 0.0;
  staf_precision = 1;
  device_id = 0;
  f_ghost = nullptr;
  f_ghost_max = 0;
}

PairSTAF::~PairSTAF()
{
  if (copymode) return;
  if (staf) {
    staf_free(staf);
    staf = nullptr;
  }
  delete[] model_dir;
  memory->destroy(f_ghost);
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

int PairSTAF::resolve_device_id() const
{
  const char *env = getenv("STAF_DEVICE_ID");
  if (env && env[0]) return utils::inumeric(FLERR, env, false, lmp);

  env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (!env) env = getenv("MPI_LOCALRANKID");
  if (!env) env = getenv("SLURM_LOCALID");
  if (!env) env = getenv("MV2_COMM_WORLD_LOCAL_RANK");
  if (env && env[0]) return atoi(env);

  /* Fallback: global rank (ok for 1 GPU/node or single-rank). */
  return comm->me;
}

void PairSTAF::settings(int narg, char **arg)
{
  if (narg < 2) error->all(FLERR, "Illegal pair_style staf command");
  cut_radial = utils::numeric(FLERR, arg[0], false, lmp);
  cut_angular = utils::numeric(FLERR, arg[1], false, lmp);
  staf_precision = 1;
  for (int i = 2; i < narg; ++i) {
    if (strcmp(arg[i], "double") == 0) staf_precision = 1;
    else if (strcmp(arg[i], "float") == 0) staf_precision = 0;
  }
}

void PairSTAF::coeff(int narg, char **arg)
{
  if (narg < 3) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  delete[] model_dir;
  model_dir = utils::strdup(arg[2]);

  device_id = resolve_device_id();

  StafOptions opt;
  staf_options_default(&opt);
  opt.precision = staf_precision;
  opt.device_id = device_id;
  opt.mlp_backend = STAF_MLP_ORT;

  if (staf) staf_free(staf);
  staf = staf_load(model_dir, &opt);
  if (!staf) error->all(FLERR, "pair_staf: staf_load failed");

  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++) setflag[i][j] = 1;
}

void PairSTAF::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;
}

void PairSTAF::init_style()
{
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style staf requires newton pair on");

  /* Full list including ghosts as neighbors of owned centers (1× cutoff DD). */
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);

  const double rcut = (cut_radial > cut_angular) ? cut_radial : cut_angular;
  if (comm->cutghostuser > 0.0 && comm->cutghostuser + 1.0e-12 < rcut)
    error->all(FLERR,
               "pair_staf: comm_modify cutoff must be >= max(rcut_r, rcut_a)");
  if (comm->cutghostuser <= 0.0)
    error->warning(FLERR,
                   "pair_staf: set 'comm_modify cutoff <rcut>' "
                   "(ghost cutoff must cover STAF cutoffs)");
}

double PairSTAF::init_one(int /*i*/, int /*j*/)
{
  return (cut_radial > cut_angular) ? cut_radial : cut_angular;
}

void PairSTAF::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (!staf) error->all(FLERR, "pair_staf: model not loaded");
  if (!list) error->all(FLERR, "pair_staf: neighbor list not built");

  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;

  if (atom->nmax > f_ghost_max) {
    memory->destroy(f_ghost);
    f_ghost_max = atom->nmax;
    memory->create(f_ghost, f_ghost_max * 3, "pair:staf:f_ghost");
  }

  std::vector<double> xpos(nall * 3), fbuf(nall * 3, 0.0), vir(6, 0.0);
  std::vector<int> tbuf(nall);
  for (int i = 0; i < nall; ++i) {
    xpos[3 * i] = x[i][0];
    xpos[3 * i + 1] = x[i][1];
    xpos[3 * i + 2] = x[i][2];
    tbuf[i] = type[i] - 1;
  }

  /* Build LAMMPS-order interaction map for owned centers.
   * Filter to STAF cutoff (not skin) and sort by distance — matches JMD ime. */
  const double rcut =
      (cut_radial > cut_angular) ? cut_radial : cut_angular;
  const double rcutsq = rcut * rcut;

  int maxneigh = 0;
  for (int i = 0; i < nlocal; ++i)
    if (list->numneigh[i] > maxneigh) maxneigh = list->numneigh[i];
  if (maxneigh < 1) maxneigh = 1;

  std::vector<int> howmany(nlocal, 0);
  std::vector<int> with((size_t)nlocal * (size_t)maxneigh, 0);
  std::vector<std::pair<double, int>> neigh_buf;
  neigh_buf.reserve((size_t)maxneigh);

  for (int i = 0; i < nlocal; ++i) {
    int *jlist = list->firstneigh[i];
    int jnum = list->numneigh[i];
    const double xi = x[i][0], yi = x[i][1], zi = x[i][2];
    neigh_buf.clear();
    for (int jj = 0; jj < jnum; ++jj) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      if (j < 0 || j >= nall || j == i) continue;
      double dx = xi - x[j][0];
      double dy = yi - x[j][1];
      double dz = zi - x[j][2];
      double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq > rcutsq || rsq < 1.0e-24) continue;
      neigh_buf.emplace_back(rsq, j);
    }
    std::sort(neigh_buf.begin(), neigh_buf.end(),
              [](const std::pair<double, int> &a,
                 const std::pair<double, int> &b) { return a.first < b.first; });
    int nn = (int)neigh_buf.size();
    if (nn > maxneigh) nn = maxneigh;
    howmany[i] = nn;
    for (int k = 0; k < nn; ++k)
      with[(size_t)i * (size_t)maxneigh + (size_t)k] = neigh_buf[k].second;
  }

  double box6[6];
  if (domain->triclinic)
    error->all(FLERR, "pair_staf: triclinic boxes not yet supported");
  box6[0] = domain->boxhi[0] - domain->boxlo[0];
  box6[1] = 0.0;
  box6[2] = 0.0;
  box6[3] = domain->boxhi[1] - domain->boxlo[1];
  box6[4] = 0.0;
  box6[5] = domain->boxhi[2] - domain->boxlo[2];

  double e_rank = 0.0;
  int rc = staf_compute(staf, nlocal, nghost, xpos.data(), box6, tbuf.data(),
                        howmany.data(), with.data(), maxneigh, &e_rank,
                        fbuf.data(), vir.data());
  if (rc != 0) error->all(FLERR, "pair_staf: staf_compute failed");

  /* Owned forces → atom->f; ghost forces → scratch then reverse_comm. */
  for (int i = 0; i < nlocal; ++i) {
    f[i][0] += fbuf[3 * i];
    f[i][1] += fbuf[3 * i + 1];
    f[i][2] += fbuf[3 * i + 2];
  }
  for (int i = nlocal; i < nall; ++i) {
    f_ghost[3 * i] = fbuf[3 * i];
    f_ghost[3 * i + 1] = fbuf[3 * i + 1];
    f_ghost[3 * i + 2] = fbuf[3 * i + 2];
  }

  if (force->newton_pair) comm->reverse_comm(this);

  if (eflag_global) eng_vdwl += e_rank;
  if (vflag_global) {
    for (int i = 0; i < 6; ++i) virial[i] += vir[i];
  }
}

int PairSTAF::pack_reverse_comm(int n, int first, double *buf)
{
  int m = 0;
  int last = first + n;
  for (int i = first; i < last; ++i) {
    buf[m++] = f_ghost[3 * i];
    buf[m++] = f_ghost[3 * i + 1];
    buf[m++] = f_ghost[3 * i + 2];
  }
  return m;
}

void PairSTAF::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m = 0;
  double **f = atom->f;
  for (int i = 0; i < n; ++i) {
    int j = list[i];
    f[j][0] += buf[m++];
    f[j][1] += buf[m++];
    f[j][2] += buf[m++];
  }
}
