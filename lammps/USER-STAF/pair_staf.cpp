/* -*- c++ -*- ----------------------------------------------------------
   USER-STAF — pair_style staf (B1 smoke: single-rank via libstaf+JMD)
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

#include <cstring>
#include <cmath>
#include <vector>

#include "staf.h"

using namespace LAMMPS_NS;
using namespace NeighConst;

PairSTAF::PairSTAF(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;

  model_dir = nullptr;
  staf = nullptr;
  cut_radial = cut_angular = 0.0;
  staf_precision = 1;
}

PairSTAF::~PairSTAF()
{
  if (copymode) return;
  if (staf) {
    staf_free(staf);
    staf = nullptr;
  }
  delete[] model_dir;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
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

  StafOptions opt;
  staf_options_default(&opt);
  opt.precision = staf_precision;
  opt.device_id = 0;
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

  neighbor->add_request(this, REQ_FULL | REQ_GHOST);

  if (comm->cutghostuser > 0.0 && comm->cutghostuser < cut_angular)
    error->warning(FLERR, "pair_staf: comm cutoff < angular cutoff");
}

double PairSTAF::init_one(int /*i*/, int /*j*/)
{
  return (cut_radial > cut_angular) ? cut_radial : cut_angular;
}

void PairSTAF::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (!staf) error->all(FLERR, "pair_staf: model not loaded");

  /* B1 smoke: single-rank, all atoms local (no ghost DD yet). */
  if (comm->nprocs > 1)
    error->all(FLERR, "pair_staf B1 smoke supports 1 MPI rank only");

  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;

  /* Pack positions; map LAMMPS types 1..ntypes → 0.. for libstaf (unused in JMD typemap). */
  std::vector<double> xpos(nall * 3), fbuf(nall * 3, 0.0), vir(6, 0.0);
  std::vector<int> tbuf(nall);
  for (int i = 0; i < nall; ++i) {
    xpos[3 * i] = x[i][0];
    xpos[3 * i + 1] = x[i][1];
    xpos[3 * i + 2] = x[i][2];
    tbuf[i] = type[i] - 1;
  }

  /* Orthorhombic box from domain */
  double box6[6];
  box6[0] = domain->boxhi[0] - domain->boxlo[0];
  box6[1] = 0.0;
  box6[2] = 0.0;
  box6[3] = domain->boxhi[1] - domain->boxlo[1];
  box6[4] = 0.0;
  box6[5] = domain->boxhi[2] - domain->boxlo[2];

  double e_rank = 0.0;
  int rc = staf_compute(staf, nlocal, nghost, xpos.data(), box6, tbuf.data(),
                        &e_rank, fbuf.data(), vir.data());
  if (rc != 0) error->all(FLERR, "pair_staf: staf_compute failed");

  for (int i = 0; i < nlocal; ++i) {
    f[i][0] += fbuf[3 * i];
    f[i][1] += fbuf[3 * i + 1];
    f[i][2] += fbuf[3 * i + 2];
  }

  if (eflag_global) eng_vdwl += e_rank;
  if (vflag_global) {
    for (int i = 0; i < 6; ++i) virial[i] += vir[i];
  }
}
