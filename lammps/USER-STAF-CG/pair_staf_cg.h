/* -*- c++ -*- ----------------------------------------------------------
   USER-STAF-CG — pair_style staf/cg
   Runtime: libstaf_cg (ORT MLP + CUDA AF/force, dual cutoff / color maps).
   Optional WCA: radial (inter non-sticky) + angular (sticky-vertex OP).
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(staf/cg, PairSTAFCG);
// clang-format on
#else

#ifndef LMP_PAIR_STAF_CG_H
#define LMP_PAIR_STAF_CG_H

#include "pair.h"

#include <vector>

struct StafModel; /* from libstaf_cg/include/staf.h */

namespace LAMMPS_NS {

class PairSTAFCG : public Pair {
 public:
  PairSTAFCG(class LAMMPS *);
  ~PairSTAFCG() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 protected:
  char *model_dir;
  double cut_radial, cut_angular;
  StafModel *staf;
  int staf_precision; /* 0 float, 1 double */
  int device_id;

  /* Scratch for reverse_comm of ghost forces before summing into atom->f. */
  double *f_ghost; /* [nmax*3], only ghosts used */
  int f_ghost_max;

  /* Radial WCA on inter non-sticky pairs (JMD ch_type==1). Off if eps<=0. */
  double wca_sigma, wca_eps, wca_cut;
  /* Radial WCA on sticky–sticky (ch_type==2). Off if eps<=0 (JMD default). */
  double wca_sticky_sigma, wca_sticky_eps, wca_sticky_cut;

  /* Radial WCA on the 4 sticky–opposite-surface "hinge" pairs. Off if eps<=0. */
  double wca_hinge_sigma, wca_hinge_eps, wca_hinge_cut;

  /* Angular WCA on each sticky-vertex side angle (degrees). Off if eps<=0. */
  double wca_ang_sigma, wca_ang_eps, wca_ang_cut, wca_ang_rmax;

  /* Bead maps from the model dir (0-based bead id = tag-1). */
  std::vector<int> map_intra, color_type_map, map_color_interaction;
  int n_beads;

  /* Origami dimer angular triplets: vertex, other-sticky, surface (1-based tags). */
  struct AngTrip {
    int vertex, sticky, surface;
  };
  std::vector<AngTrip> ang_trips;

  /* Hinge pairs: 1-based tags (surface of one origami, sticky of the other). */
  struct HingePair {
    int a, b;
  };
  std::vector<HingePair> hinge_pairs;

  void allocate();
  int resolve_device_id() const;
  double cut_max() const;
  void load_wca_maps();
  void setup_ang_trips();
  void setup_hinge_pairs();
  void compute_wca_radial(int eflag);
  void compute_wca_angular(int eflag);
  void compute_wca_hinge(int eflag);
  void add_force(int i, double fx, double fy, double fz);
  int find_closest_tag(int from, int want_tag) const;
  int bead_id(int i) const;
  int pair_ch_type(int bi, int bj) const;
};

}  // namespace LAMMPS_NS

#endif
#endif
