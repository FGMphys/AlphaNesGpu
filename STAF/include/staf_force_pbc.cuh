#ifndef STAF_FORCE_PBC_CUH
#define STAF_FORCE_PBC_CUH

/* Match libstaf/jmd: when 0, apply minimum-image rint on fractional coords. */
#ifdef STAF_FORCE_PBC_DEFINE
__device__ __constant__ int staf_force_skip_pbc = 0;
#else
extern __device__ __constant__ int staf_force_skip_pbc;
#endif

/*
 * Cartesian → fractional (same inverse-box as descriptor convert_carte_to_int /
 * libstaf staf_api). STAF box6 = [Lx, xy, xz, Ly, yz, Lz].
 * Virial kernels must convert before rint MIC: TF passes Cartesian pos.
 */
__device__ inline void staf_cart_to_frac(real px, real py, real pz,
                                        const real* box_b, real& fx, real& fy,
                                        real& fz) {
  const real i0 = real(1.) / box_b[0];
  const real i1 = -box_b[1] / (box_b[0] * box_b[3]);
  const real i2 = (box_b[1] * box_b[4]) / (box_b[0] * box_b[3] * box_b[5]) -
                  box_b[2] / (box_b[0] * box_b[5]);
  const real i3 = real(1.) / box_b[3];
  const real i4 = -box_b[4] / (box_b[3] * box_b[5]);
  const real i5 = real(1.) / box_b[5];
  fx = i0 * px + i1 * py + i2 * pz;
  fy = i3 * py + i4 * pz;
  fz = i5 * pz;
}

/* Fractional MIC then back to Cartesian separation (jmd force virial path). */
__device__ inline void staf_min_image_cart_from_cart(
    real pix, real piy, real piz, real pjx, real pjy, real pjz,
    const real* box_b, real& dx, real& dy, real& dz) {
  real fix, fiy, fiz, fjx, fjy, fjz;
  staf_cart_to_frac(pix, piy, piz, box_b, fix, fiy, fiz);
  staf_cart_to_frac(pjx, pjy, pjz, box_b, fjx, fjy, fjz);
  real rijx = fix - fjx;
  real rijy = fiy - fjy;
  real rijz = fiz - fjz;
  if (!staf_force_skip_pbc) {
    rijx -= rint(rijx);
    rijy -= rint(rijy);
    rijz -= rint(rijz);
  }
  dx = box_b[0] * rijx + box_b[1] * rijy + box_b[2] * rijz;
  dy = box_b[3] * rijy + box_b[4] * rijz;
  dz = box_b[5] * rijz;
}

#endif /* STAF_FORCE_PBC_CUH */
