/* -*- c++ -*- ----------------------------------------------------------
   Optional FixSTAF — per-rank GPU pin / model lifetime helpers (scaffold)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(staf, FixSTAF);
// clang-format on
#else

#ifndef LMP_FIX_STAF_H
#define LMP_FIX_STAF_H

#include "fix.h"

namespace LAMMPS_NS {

class FixSTAF : public Fix {
 public:
  FixSTAF(class LAMMPS *, int, char **);
  int setmask() override;
  void init() override;

 protected:
  int device_id;
};

}  // namespace LAMMPS_NS

#endif
#endif
