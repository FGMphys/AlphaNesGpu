/* -*- c++ -*- ----------------------------------------------------------
   FixSTAF — pin CUDA device from MPI local rank (or explicit id)
------------------------------------------------------------------------- */

#include "fix_staf.h"

#include "comm.h"
#include "error.h"
#include "utils.h"

#include "staf.h"

using namespace LAMMPS_NS;

FixSTAF::FixSTAF(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  if (narg < 4) error->all(FLERR, "Illegal fix staf command");
  device_id = utils::inumeric(FLERR, arg[3], false, lmp);
}

int FixSTAF::setmask()
{
  return 0;
}

void FixSTAF::init()
{
  if (staf_cuda_set_device(device_id) != 0)
    error->all(FLERR, "fix staf: staf_cuda_set_device failed");
  if (comm->me == 0)
    utils::logmesg(lmp, "Fix staf: pinned CUDA device {}\n", device_id);
}
