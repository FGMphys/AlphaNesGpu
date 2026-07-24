/* -*- c++ -*- ----------------------------------------------------------
   FixSTAF scaffold — pin CUDA device from MPI local rank (optional)
------------------------------------------------------------------------- */

#include "fix_staf.h"

#include "comm.h"
#include "error.h"

using namespace LAMMPS_NS;

FixSTAF::FixSTAF(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  if (narg < 3) error->all(FLERR, "Illegal fix staf command");
  device_id = utils::inumeric(FLERR, arg[3], false, lmp);
}

int FixSTAF::setmask()
{
  return 0;
}

void FixSTAF::init()
{
  /* TODO: cudaSetDevice(device_id) or map from comm->me % ngpus */
  if (comm->me == 0)
    error->warning(FLERR, "fix staf: scaffold only (no CUDA pin yet)");
}
