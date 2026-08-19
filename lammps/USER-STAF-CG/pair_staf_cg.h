/* -*- c++ -*- ----------------------------------------------------------
   USER-STAF-CG — pair_style staf/cg
   Runtime: libstaf_cg (ORT MLP + CUDA AF/force, dual cutoff / color maps).
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(staf/cg, PairSTAFCG);
// clang-format on
#else

#ifndef LMP_PAIR_STAF_CG_H
#define LMP_PAIR_STAF_CG_H

#include "pair.h"

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

  void allocate();
  int resolve_device_id() const;
};

}  // namespace LAMMPS_NS

#endif
#endif
