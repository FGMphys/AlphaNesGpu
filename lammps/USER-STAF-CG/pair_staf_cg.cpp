/* -*- c++ -*- ----------------------------------------------------------
   USER-STAF-CG — pair_style staf/cg (Allegro-style DD via LAMMPS neigh + reverse_comm)
   Optional WCA: radial (inter non-sticky) + angular (sticky-vertex OP).
------------------------------------------------------------------------- */

#include "pair_staf_cg.h"

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
#include <cstdio>
#include <fstream>
#include <string>

#include "staf.h"

static constexpr double TWO_1_6 = 1.122462048309373; /* 2^{1/6} */
static constexpr double RAD2DEG = 57.29577951308232;
static constexpr double ANG_SMALL = 0.001;
static constexpr double THETA_DEG_MIN = 0.5;

using namespace LAMMPS_NS;
using namespace NeighConst;

PairSTAFCG::PairSTAFCG(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  /* Ghost forces from many-body chain rule → reverse_comm into owned atoms. */
  comm_reverse = 3;
  /* libstaf_cg returns pair virial (diag); do NOT use virial_fdotr_compute —
     ghost force pieces live in f_ghost until reverse_comm, so FDOTR would
     miss them / double-count. */
  no_virial_fdotr_compute = 1;

  model_dir = nullptr;
  staf = nullptr;
  cut_radial = cut_angular = 50.0;
  staf_precision = 0; /* float32 ONNX default (CG) */
  device_id = 0;
  f_ghost = nullptr;
  f_ghost_max = 0;

  wca_sigma = wca_eps = wca_cut = 0.0;
  wca_sticky_sigma = wca_sticky_eps = wca_sticky_cut = 0.0;
  wca_hinge_sigma = wca_hinge_eps = wca_hinge_cut = 0.0;
  wca_ang_sigma = wca_ang_eps = wca_ang_cut = 0.0;
  wca_ang_rmax = 20.0;
  n_beads = 0;
}

PairSTAFCG::~PairSTAFCG()
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

int PairSTAFCG::resolve_device_id() const
{
  const char *env = getenv("STAF_DEVICE_ID");
  if (env && env[0]) return utils::inumeric(FLERR, env, false, lmp);

  env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (!env) env = getenv("MPI_LOCALRANKID");
  if (!env) env = getenv("SLURM_LOCALID");
  if (!env) env = getenv("MV2_COMM_WORLD_LOCAL_RANK");
  if (env && env[0]) return atoi(env);

  /* Fallback: global rank (ok for 1 GPU/node or single-rank).
   * libstaf_cg maps device_id % deviceCount, so several ranks may share a GPU. */
  return comm->me;
}

double PairSTAFCG::cut_max() const
{
  double r = (cut_radial > cut_angular) ? cut_radial : cut_angular;
  if (wca_eps > 0.0 && wca_cut > r) r = wca_cut;
  if (wca_sticky_eps > 0.0 && wca_sticky_cut > r) r = wca_sticky_cut;
  if (wca_hinge_eps > 0.0 && wca_hinge_cut > r) r = wca_hinge_cut;
  return r;
}

void PairSTAFCG::settings(int narg, char **arg)
{
  /* Defaults for origami 1-epoch export: intra Rc = 50 Å. */
  cut_radial = 50.0;
  cut_angular = 50.0;
  staf_precision = 0;
  wca_sigma = wca_eps = wca_cut = 0.0;
  wca_sticky_sigma = wca_sticky_eps = wca_sticky_cut = 0.0;
  wca_hinge_sigma = wca_hinge_eps = wca_hinge_cut = 0.0;
  wca_ang_sigma = wca_ang_eps = wca_ang_cut = 0.0;
  wca_ang_rmax = 20.0;

  if (narg == 0) return;
  if (narg < 2) error->all(FLERR, "Illegal pair_style staf/cg command");
  cut_radial = utils::numeric(FLERR, arg[0], false, lmp);
  cut_angular = utils::numeric(FLERR, arg[1], false, lmp);

  auto need_val = [&](int i, const char *key) {
    if (i + 1 >= narg) {
      char msg[256];
      snprintf(msg, sizeof(msg), "pair_style staf/cg: missing value for %s", key);
      error->all(FLERR, msg);
    }
  };

  for (int i = 2; i < narg; ++i) {
    if (strcmp(arg[i], "double") == 0) staf_precision = 1;
    else if (strcmp(arg[i], "float") == 0) staf_precision = 0;
    else if (strcmp(arg[i], "wca_sigma") == 0) {
      need_val(i, arg[i]);
      wca_sigma = utils::numeric(FLERR, arg[++i], false, lmp);
    } else if (strcmp(arg[i], "wca_eps") == 0) {
      need_val(i, arg[i]);
      wca_eps = utils::numeric(FLERR, arg[++i], false, lmp);
    } else if (strcmp(arg[i], "wca_cut") == 0) {
      need_val(i, arg[i]);
      wca_cut = utils::numeric(FLERR, arg[++i], false, lmp);
    } else if (strcmp(arg[i], "wca_sticky_sigma") == 0) {
      need_val(i, arg[i]);
      wca_sticky_sigma = utils::numeric(FLERR, arg[++i], false, lmp);
    } else if (strcmp(arg[i], "wca_sticky_eps") == 0) {
      need_val(i, arg[i]);
      wca_sticky_eps = utils::numeric(FLERR, arg[++i], false, lmp);
    } else if (strcmp(arg[i], "wca_sticky_cut") == 0) {
      need_val(i, arg[i]);
      wca_sticky_cut = utils::numeric(FLERR, arg[++i], false, lmp);
    } else if (strcmp(arg[i], "wca_hinge_sigma") == 0) {
      need_val(i, arg[i]);
      wca_hinge_sigma = utils::numeric(FLERR, arg[++i], false, lmp);
    } else if (strcmp(arg[i], "wca_hinge_eps") == 0) {
      need_val(i, arg[i]);
      wca_hinge_eps = utils::numeric(FLERR, arg[++i], false, lmp);
    } else if (strcmp(arg[i], "wca_hinge_cut") == 0) {
      need_val(i, arg[i]);
      wca_hinge_cut = utils::numeric(FLERR, arg[++i], false, lmp);
    } else if (strcmp(arg[i], "wca_ang_sigma") == 0) {
      need_val(i, arg[i]);
      wca_ang_sigma = utils::numeric(FLERR, arg[++i], false, lmp);
    } else if (strcmp(arg[i], "wca_ang_eps") == 0) {
      need_val(i, arg[i]);
      wca_ang_eps = utils::numeric(FLERR, arg[++i], false, lmp);
    } else if (strcmp(arg[i], "wca_ang_cut") == 0) {
      need_val(i, arg[i]);
      wca_ang_cut = utils::numeric(FLERR, arg[++i], false, lmp);
    } else if (strcmp(arg[i], "wca_ang_rmax") == 0) {
      need_val(i, arg[i]);
      wca_ang_rmax = utils::numeric(FLERR, arg[++i], false, lmp);
    } else {
      char msg[256];
      snprintf(msg, sizeof(msg), "Unknown pair_style staf/cg keyword: %s", arg[i]);
      error->all(FLERR, msg);
    }
  }

  if (wca_sigma < 0.0 || wca_eps < 0.0 || wca_cut < 0.0)
    error->all(FLERR, "pair_staf/cg: WCA radial parameters must be >= 0");
  if (wca_sticky_sigma < 0.0 || wca_sticky_eps < 0.0 || wca_sticky_cut < 0.0)
    error->all(FLERR, "pair_staf/cg: WCA sticky parameters must be >= 0");
  if (wca_hinge_sigma < 0.0 || wca_hinge_eps < 0.0 || wca_hinge_cut < 0.0)
    error->all(FLERR, "pair_staf/cg: WCA hinge parameters must be >= 0");
  if (wca_ang_sigma < 0.0 || wca_ang_eps < 0.0 || wca_ang_cut < 0.0)
    error->all(FLERR, "pair_staf/cg: WCA angular parameters must be >= 0");
  if ((wca_sigma > 0.0) != (wca_eps > 0.0))
    error->all(FLERR, "pair_staf/cg: set both wca_sigma and wca_eps (or neither)");
  if ((wca_sticky_sigma > 0.0) != (wca_sticky_eps > 0.0))
    error->all(FLERR, "pair_staf/cg: set both wca_sticky_sigma and wca_sticky_eps (or neither)");
  if ((wca_hinge_sigma > 0.0) != (wca_hinge_eps > 0.0))
    error->all(FLERR, "pair_staf/cg: set both wca_hinge_sigma and wca_hinge_eps (or neither)");
  if ((wca_ang_sigma > 0.0) != (wca_ang_eps > 0.0))
    error->all(FLERR, "pair_staf/cg: set both wca_ang_sigma and wca_ang_eps (or neither)");

  if (wca_eps > 0.0 && wca_cut <= 0.0) wca_cut = TWO_1_6 * wca_sigma;
  if (wca_sticky_eps > 0.0 && wca_sticky_cut <= 0.0)
    wca_sticky_cut = TWO_1_6 * wca_sticky_sigma;
  if (wca_hinge_eps > 0.0 && wca_hinge_cut <= 0.0)
    wca_hinge_cut = TWO_1_6 * wca_hinge_sigma;
  if (wca_ang_eps > 0.0 && wca_ang_cut <= 0.0) wca_ang_cut = TWO_1_6 * wca_ang_sigma;
}

void PairSTAFCG::coeff(int narg, char **arg)
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
  if (!staf) error->all(FLERR, "pair_staf/cg: staf_load failed");

  load_wca_maps();
  setup_ang_trips();
  setup_hinge_pairs();

  if (comm->me == 0) {
    if (wca_eps > 0.0) {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "pair_staf/cg WCA radial: sigma=%g eps=%g cut=%g (inter non-sticky)\n",
               wca_sigma, wca_eps, wca_cut);
      if (screen) fputs(msg, screen);
      if (logfile) fputs(msg, logfile);
    }
    if (wca_sticky_eps > 0.0) {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "pair_staf/cg WCA sticky: sigma=%g eps=%g cut=%g (sticky-sticky)\n",
               wca_sticky_sigma, wca_sticky_eps, wca_sticky_cut);
      if (screen) fputs(msg, screen);
      if (logfile) fputs(msg, logfile);
    }
    if (wca_hinge_eps > 0.0) {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "pair_staf/cg WCA hinge: sigma=%g eps=%g cut=%g  n_pairs=%d\n",
               wca_hinge_sigma, wca_hinge_eps, wca_hinge_cut,
               (int)hinge_pairs.size());
      if (screen) fputs(msg, screen);
      if (logfile) fputs(msg, logfile);
    }
    if (wca_ang_eps > 0.0) {
      char msg[320];
      snprintf(msg, sizeof(msg),
               "pair_staf/cg WCA angular: sigma=%g deg  eps=%g  cut=%g deg  "
               "rmax=%g A  n_triplets=%d\n",
               wca_ang_sigma, wca_ang_eps, wca_ang_cut, wca_ang_rmax,
               (int)ang_trips.size());
      if (screen) fputs(msg, screen);
      if (logfile) fputs(msg, logfile);
    }
  }

  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++) setflag[i][j] = 1;
}

void PairSTAFCG::load_wca_maps()
{
  map_intra.clear();
  color_type_map.clear();
  map_color_interaction.clear();
  n_beads = 0;
  if (wca_eps <= 0.0 && wca_sticky_eps <= 0.0) return;

  auto load_ints = [&](const char *name, std::vector<int> &out) {
    std::string path = std::string(model_dir) + "/" + name;
    std::ifstream in(path.c_str());
    if (!in) {
      char msg[512];
      snprintf(msg, sizeof(msg), "pair_staf/cg: cannot read %s", path.c_str());
      error->all(FLERR, msg);
    }
    out.clear();
    double x;
    while (in >> x) out.push_back((int)std::lround(x));
    if (out.empty()) {
      char msg[512];
      snprintf(msg, sizeof(msg), "pair_staf/cg: empty file %s", path.c_str());
      error->all(FLERR, msg);
    }
  };

  load_ints("map_intra.dat", map_intra);
  load_ints("color_type_map.dat", color_type_map);
  load_ints("map_color_interaction.dat", map_color_interaction);
  if ((int)map_intra.size() != (int)color_type_map.size())
    error->all(FLERR, "pair_staf/cg: map_intra and color_type_map length mismatch");
  n_beads = (int)map_intra.size();
}

void PairSTAFCG::setup_ang_trips()
{
  ang_trips.clear();
  if (wca_ang_eps <= 0.0) return;

  /* 24-bead dimer, 1-based tags. Same topology as origami_op.py:
   * SURFACE_1 = [2,9,1,11,6], SURFACE_2 = [21,22,18,23,16]
   * sticky vertices tags 10 and 17. */
  const int surf1[] = {2, 9, 1, 11, 6};
  const int surf2[] = {21, 22, 18, 23, 16};
  const int v1 = 10, v2 = 17;
  for (int k = 0; k < 5; ++k)
    ang_trips.push_back(AngTrip{v1, v2, surf1[k]});
  for (int k = 0; k < 5; ++k)
    ang_trips.push_back(AngTrip{v2, v1, surf2[k]});
}

void PairSTAFCG::setup_hinge_pairs()
{
  hinge_pairs.clear();
  if (wca_hinge_eps <= 0.0) return;
  /* 1-based tags. Same four pairs as score_run.py HINGE:
   * surface of origami 1 — sticky of origami 2, and vice versa.
   * These fill g(r) at ~18–22 Å in collapsed CG, vs ~32 Å in AA. */
  hinge_pairs.push_back(HingePair{9, 17});
  hinge_pairs.push_back(HingePair{11, 17});
  hinge_pairs.push_back(HingePair{10, 21});
  hinge_pairs.push_back(HingePair{10, 22});
}

void PairSTAFCG::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;
}

void PairSTAFCG::init_style()
{
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style staf/cg requires newton pair on");

  if ((wca_ang_eps > 0.0 || wca_hinge_eps > 0.0) && atom->natoms != 24)
    error->all(FLERR,
               "pair_staf/cg: angular/hinge WCA is hardcoded for the 24-bead origami dimer");

  /* Full list including ghosts as neighbors of owned centers (1× cutoff DD). */
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);

  const double rcut = cut_max();
  if (comm->cutghostuser > 0.0 && comm->cutghostuser + 1.0e-12 < rcut)
    error->all(FLERR,
               "pair_staf/cg: comm_modify cutoff must be >= max(rcut_r, rcut_a) "
               "(origami intra Rc is typically 50 Å)");
  if (comm->cutghostuser <= 0.0)
    error->warning(FLERR,
                   "pair_staf/cg: set 'comm_modify cutoff <rcut>' "
                   "(ghost cutoff must cover STAF-CG cutoffs; intra Rc=50)");
}

double PairSTAFCG::init_one(int /*i*/, int /*j*/)
{
  return cut_max();
}

void PairSTAFCG::add_force(int i, double fx, double fy, double fz)
{
  if (i < atom->nlocal) {
    atom->f[i][0] += fx;
    atom->f[i][1] += fy;
    atom->f[i][2] += fz;
  } else {
    f_ghost[3 * i] += fx;
    f_ghost[3 * i + 1] += fy;
    f_ghost[3 * i + 2] += fz;
  }
}

int PairSTAFCG::bead_id(int i) const
{
  int bid = (int)atom->tag[i] - 1;
  if (bid < 0 || bid >= n_beads)
    error->one(FLERR, "pair_staf/cg: atom tag outside map_intra range");
  return bid;
}

int PairSTAFCG::pair_ch_type(int bi, int bj) const
{
  /* 0 intra, 2 sticky-sticky, 1 inter non-sticky (WCA). Same as JMD. */
  if (map_intra[bi] == map_intra[bj]) return 0;
  int coli = color_type_map[bi];
  int colj = color_type_map[bj];
  if (coli < 0 || coli >= (int)map_color_interaction.size())
    error->one(FLERR, "pair_staf/cg: color index out of range");
  if (map_color_interaction[coli] == colj) return 2;
  return 1;
}

int PairSTAFCG::find_closest_tag(int from, int want_tag) const
{
  double **x = atom->x;
  tagint *tag = atom->tag;
  const int nall = atom->nlocal + atom->nghost;
  int best = -1;
  double best_rsq = 1.0e300;
  for (int i = 0; i < nall; ++i) {
    if ((int)tag[i] != want_tag) continue;
    double dx = x[i][0] - x[from][0];
    double dy = x[i][1] - x[from][1];
    double dz = x[i][2] - x[from][2];
    domain->minimum_image(dx, dy, dz);
    double rsq = dx * dx + dy * dy + dz * dz;
    if (rsq < best_rsq) {
      best_rsq = rsq;
      best = i;
    }
  }
  return best;
}

void PairSTAFCG::compute_wca_radial(int eflag)
{
  if (wca_eps <= 0.0 && wca_sticky_eps <= 0.0) return;

  double **x = atom->x;
  tagint *tag = atom->tag;
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;

  for (int i = 0; i < nlocal; ++i) {
    int *jlist = list->firstneigh[i];
    int jnum = list->numneigh[i];
    const double xi = x[i][0], yi = x[i][1], zi = x[i][2];
    const int ti = (int)tag[i];
    const int bi = bead_id(i);
    for (int jj = 0; jj < jnum; ++jj) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      if (j < 0 || j >= nall || j == i) continue;
      if ((int)tag[j] <= ti) continue; /* each pair once */
      const int ch = pair_ch_type(bi, bead_id(j));
      double sigma, eps, cut;
      if (ch == 1) {
        if (wca_eps <= 0.0) continue;
        sigma = wca_sigma;
        eps = wca_eps;
        cut = wca_cut;
      } else if (ch == 2) {
        if (wca_sticky_eps <= 0.0) continue;
        sigma = wca_sticky_sigma;
        eps = wca_sticky_eps;
        cut = wca_sticky_cut;
      } else {
        continue;
      }

      double dx = xi - x[j][0];
      double dy = yi - x[j][1];
      double dz = zi - x[j][2];
      domain->minimum_image(dx, dy, dz);
      double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq >= cut * cut || rsq < 1.0e-24) continue;

      double r2inv = 1.0 / rsq;
      double r6inv = r2inv * r2inv * r2inv;
      double sig2 = sigma * sigma;
      double sig6 = sig2 * sig2 * sig2;
      double s6 = sig6 * r6inv;
      double s12 = s6 * s6;
      double forcelj = 24.0 * eps * (2.0 * s12 - s6);
      double fpair = forcelj * r2inv;

      add_force(i, fpair * dx, fpair * dy, fpair * dz);
      add_force(j, -fpair * dx, -fpair * dy, -fpair * dz);

      if (eflag && eflag_global) eng_vdwl += 4.0 * eps * (s12 - s6) + eps;
      if (vflag_global) {
        virial[0] += fpair * dx * dx;
        virial[1] += fpair * dy * dy;
        virial[2] += fpair * dz * dz;
        virial[3] += fpair * dx * dy;
        virial[4] += fpair * dx * dz;
        virial[5] += fpair * dy * dz;
      }
    }
  }
}

void PairSTAFCG::compute_wca_hinge(int eflag)
{
  if (wca_hinge_eps <= 0.0) return;

  double **x = atom->x;
  tagint *tag = atom->tag;
  const int nlocal = atom->nlocal;
  const double sigma = wca_hinge_sigma;
  const double eps = wca_hinge_eps;
  const double cut = wca_hinge_cut;
  const double cutsq = cut * cut;

  for (const HingePair &hp : hinge_pairs) {
    int i = -1;
    for (int k = 0; k < nlocal; ++k) {
      if ((int)tag[k] == hp.a) {
        i = k;
        break;
      }
    }
    if (i < 0) continue;
    int j = find_closest_tag(i, hp.b);
    if (j < 0) continue;

    double dx = x[i][0] - x[j][0];
    double dy = x[i][1] - x[j][1];
    double dz = x[i][2] - x[j][2];
    domain->minimum_image(dx, dy, dz);
    double rsq = dx * dx + dy * dy + dz * dz;
    if (rsq >= cutsq || rsq < 1.0e-24) continue;

    double r2inv = 1.0 / rsq;
    double r6inv = r2inv * r2inv * r2inv;
    double sig2 = sigma * sigma;
    double sig6 = sig2 * sig2 * sig2;
    double s6 = sig6 * r6inv;
    double s12 = s6 * s6;
    double forcelj = 24.0 * eps * (2.0 * s12 - s6);
    double fpair = forcelj * r2inv;

    add_force(i, fpair * dx, fpair * dy, fpair * dz);
    add_force(j, -fpair * dx, -fpair * dy, -fpair * dz);

    if (eflag && eflag_global) eng_vdwl += 4.0 * eps * (s12 - s6) + eps;
    if (vflag_global) {
      virial[0] += fpair * dx * dx;
      virial[1] += fpair * dy * dy;
      virial[2] += fpair * dz * dz;
      virial[3] += fpair * dx * dy;
      virial[4] += fpair * dx * dz;
      virial[5] += fpair * dy * dz;
    }
  }
}

void PairSTAFCG::compute_wca_angular(int eflag)
{
  if (wca_ang_eps <= 0.0) return;

  double **x = atom->x;
  const int nlocal = atom->nlocal;
  const double rmaxsq =
      (wca_ang_rmax > 0.0) ? wca_ang_rmax * wca_ang_rmax : 1.0e300;

  tagint *tag = atom->tag;
  for (const AngTrip &t : ang_trips) {
    /* Vertex must be owned to count the angle once (PBC ghosts ignored). */
    int i2 = -1;
    for (int i = 0; i < nlocal; ++i) {
      if ((int)tag[i] == t.vertex) {
        i2 = i;
        break;
      }
    }
    if (i2 < 0) continue;
    int i1 = find_closest_tag(i2, t.sticky);
    int i3 = find_closest_tag(i2, t.surface);
    if (i1 < 0 || i3 < 0) continue;

    double delx1 = x[i1][0] - x[i2][0];
    double dely1 = x[i1][1] - x[i2][1];
    double delz1 = x[i1][2] - x[i2][2];
    domain->minimum_image(delx1, dely1, delz1);
    double rsq1 = delx1 * delx1 + dely1 * dely1 + delz1 * delz1;

    if (rsq1 >= rmaxsq) continue; /* far dimer: no angular wall */

    double delx2 = x[i3][0] - x[i2][0];
    double dely2 = x[i3][1] - x[i2][1];
    double delz2 = x[i3][2] - x[i2][2];
    domain->minimum_image(delx2, dely2, delz2);
    double rsq2 = delx2 * delx2 + dely2 * dely2 + delz2 * delz2;
    if (rsq1 < 1.0e-24 || rsq2 < 1.0e-24) continue;

    double r1 = sqrt(rsq1);
    double r2 = sqrt(rsq2);
    double c = (delx1 * delx2 + dely1 * dely2 + delz1 * delz2) / (r1 * r2);
    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;
    double s = sqrt(1.0 - c * c);
    if (s < ANG_SMALL) s = ANG_SMALL;
    s = 1.0 / s;

    double theta_deg = acos(c) * RAD2DEG;
    if (theta_deg >= wca_ang_cut) continue;
    if (theta_deg < THETA_DEG_MIN) theta_deg = THETA_DEG_MIN;

    double ratio = wca_ang_sigma / theta_deg;
    double r6 = ratio * ratio * ratio;
    r6 *= r6;
    double r12 = r6 * r6;
    double ewca = 4.0 * wca_ang_eps * (r12 - r6) + wca_ang_eps;
    /* dU/dθ_deg, then to radians. */
    double dUdth_deg =
        -24.0 * wca_ang_eps * (2.0 * r12 - r6) / theta_deg;
    double dUdth = dUdth_deg * RAD2DEG;

    double a = -dUdth * s;
    double a11 = a * c / rsq1;
    double a12 = -a / (r1 * r2);
    double a22 = a * c / rsq2;

    double f1[3], f3[3];
    f1[0] = a11 * delx1 + a12 * delx2;
    f1[1] = a11 * dely1 + a12 * dely2;
    f1[2] = a11 * delz1 + a12 * delz2;
    f3[0] = a22 * delx2 + a12 * delx1;
    f3[1] = a22 * dely2 + a12 * dely1;
    f3[2] = a22 * delz2 + a12 * delz1;

    add_force(i1, f1[0], f1[1], f1[2]);
    add_force(i2, -(f1[0] + f3[0]), -(f1[1] + f3[1]), -(f1[2] + f3[2]));
    add_force(i3, f3[0], f3[1], f3[2]);

    if (eflag && eflag_global) eng_vdwl += ewca;
    if (vflag_global) {
      virial[0] += delx1 * f1[0] + delx2 * f3[0];
      virial[1] += dely1 * f1[1] + dely2 * f3[1];
      virial[2] += delz1 * f1[2] + delz2 * f3[2];
      virial[3] += delx1 * f1[1] + delx2 * f3[1];
      virial[4] += delx1 * f1[2] + delx2 * f3[2];
      virial[5] += dely1 * f1[2] + dely2 * f3[2];
    }
  }
}

void PairSTAFCG::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (!staf) error->all(FLERR, "pair_staf/cg: model not loaded");

  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;

  if (atom->nmax > f_ghost_max) {
    memory->destroy(f_ghost);
    f_ghost_max = atom->nmax;
    memory->create(f_ghost, f_ghost_max * 3, "pair:staf_cg:f_ghost");
  }
  for (int i = 0; i < f_ghost_max * 3; ++i) f_ghost[i] = 0.0;

  /* Empty ranks (origami dimer clustered in a large box) still reverse_comm. */
  if (nlocal == 0) {
    if (force->newton_pair) comm->reverse_comm(this);
    return;
  }
  if (!list) error->all(FLERR, "pair_staf/cg: neighbor list not built");

  std::vector<double> xpos(nall * 3), fbuf(nall * 3, 0.0), vir(6, 0.0);
  std::vector<int> tbuf(nall);
  for (int i = 0; i < nall; ++i) {
    xpos[3 * i] = x[i][0];
    xpos[3 * i + 1] = x[i][1];
    xpos[3 * i + 2] = x[i][2];
    tbuf[i] = (int)tag[i] - 1;
  }

  /* Build LAMMPS-order interaction map for owned centers.
   * Filter to STAF cutoff (not skin) and sort by distance — matches JMD ime. */
  const double rcut = cut_max();
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
    error->all(FLERR, "pair_staf/cg: triclinic boxes not yet supported");
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
  if (rc != 0) error->all(FLERR, "pair_staf/cg: staf_compute failed");

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

  compute_wca_radial(eflag);
  compute_wca_angular(eflag);
  compute_wca_hinge(eflag);

  if (force->newton_pair) comm->reverse_comm(this);

  if (eflag_global) eng_vdwl += e_rank;
  if (vflag_global) {
    for (int i = 0; i < 6; ++i) virial[i] += vir[i];
  }
}

int PairSTAFCG::pack_reverse_comm(int n, int first, double *buf)
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

void PairSTAFCG::unpack_reverse_comm(int n, int *list, double *buf)
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
