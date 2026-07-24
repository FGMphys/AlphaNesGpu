/* -*- c++ -*- ----------------------------------------------------------
   USER-STAF scaffold — pair_style staf
   Runtime: libstaf (ORT MLP + CUDA AF/force). See test/B_ARCHITECTURE.md
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(staf, PairSTAF);
// clang-format on
#else

#ifndef LMP_PAIR_STAF_H
#define LMP_PAIR_STAF_H

#include "pair.h"

struct StafModel; /* from libstaf/include/staf.h */

namespace LAMMPS_NS {

class PairSTAF : public Pair {
 public:
  PairSTAF(class LAMMPS *);
  ~PairSTAF() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

 protected:
  char *model_dir;
  double cut_radial, cut_angular;
  StafModel *staf;
  int staf_precision; /* 0 float, 1 double */

  void allocate();
};

}  // namespace LAMMPS_NS

#endif
#endif
