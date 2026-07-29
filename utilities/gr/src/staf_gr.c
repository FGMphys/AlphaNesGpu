/* staf_gr.c — cell-list multi-species g(r), inspired by code_jsw_tmmc/utilities/calculate_gr.c */
#include "staf_gr.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

StafGr *staf_gr_create(double dr, double rmax) {
  if (dr <= 0.0 || rmax <= dr) return NULL;
  StafGr *gr = (StafGr *)calloc(1, sizeof(StafGr));
  if (!gr) return NULL;
  gr->dr = dr;
  gr->rmax = rmax;
  gr->nbin = (int)floor(rmax / dr);
  if (gr->nbin < 1) {
    free(gr);
    return NULL;
  }
  gr->hist = (long long *)calloc((size_t)gr->nbin, sizeof(long long));
  if (!gr->hist) {
    free(gr);
    return NULL;
  }
  gr->nsamples = 0;
  return gr;
}

void staf_gr_free(StafGr *gr) {
  if (!gr) return;
  free(gr->hist);
  free(gr);
}

void staf_gr_reset(StafGr *gr) {
  if (!gr) return;
  memset(gr->hist, 0, (size_t)gr->nbin * sizeof(long long));
  gr->nsamples = 0;
}

static int cell_id(int ix, int iy, int iz, int nx, int ny, int nz) {
  if (ix < 0) ix += nx;
  if (iy < 0) iy += ny;
  if (iz < 0) iz += nz;
  if (ix >= nx) ix -= nx;
  if (iy >= ny) iy -= ny;
  if (iz >= nz) iz -= nz;
  return ix + nx * (iy + ny * iz);
}

int staf_gr_accumulate(StafGr *gr, const double *pos, const int *types, int n,
                       const double *box, int ta, int tb) {
  if (!gr || !pos || !types || !box || n <= 0) return -1;
  const double Lx = box[0], Ly = box[1], Lz = box[2];
  if (Lx <= 0 || Ly <= 0 || Lz <= 0) return -1;
  const double invLx = 1.0 / Lx, invLy = 1.0 / Ly, invLz = 1.0 / Lz;
  const double rmax = gr->rmax;
  const double rmax2 = rmax * rmax;
  const double dr = gr->dr;
  const int nbin = gr->nbin;
  const int same = (ta == tb);

  /* cell grid */
  int nx = (int)floor(Lx / rmax);
  int ny = (int)floor(Ly / rmax);
  int nz = (int)floor(Lz / rmax);
  if (nx < 1) nx = 1;
  if (ny < 1) ny = 1;
  if (nz < 1) nz = 1;
  const int ncell = nx * ny * nz;

  int *head = (int *)malloc((size_t)ncell * sizeof(int));
  int *next = (int *)malloc((size_t)n * sizeof(int));
  int *cix = (int *)malloc((size_t)n * sizeof(int));
  int *ciy = (int *)malloc((size_t)n * sizeof(int));
  int *ciz = (int *)malloc((size_t)n * sizeof(int));
  if (!head || !next || !cix || !ciy || !ciz) {
    free(head);
    free(next);
    free(cix);
    free(ciy);
    free(ciz);
    return -1;
  }
  for (int c = 0; c < ncell; ++c) head[c] = -1;

  for (int i = 0; i < n; ++i) {
    double x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
    /* wrap into [0,L) */
    x -= Lx * floor(x * invLx);
    y -= Ly * floor(y * invLy);
    z -= Lz * floor(z * invLz);
    int ix = (int)floor(x / Lx * nx);
    int iy = (int)floor(y / Ly * ny);
    int iz = (int)floor(z / Lz * nz);
    if (ix < 0) ix = 0;
    if (iy < 0) iy = 0;
    if (iz < 0) iz = 0;
    if (ix >= nx) ix = nx - 1;
    if (iy >= ny) iy = ny - 1;
    if (iz >= nz) iz = nz - 1;
    cix[i] = ix;
    ciy[i] = iy;
    ciz[i] = iz;
    int cid = cell_id(ix, iy, iz, nx, ny, nz);
    next[i] = head[cid];
    head[cid] = i;
  }

  for (int i = 0; i < n; ++i) {
    if (types[i] != ta) continue;
    const double xi = pos[3 * i], yi = pos[3 * i + 1], zi = pos[3 * i + 2];
    const int ix0 = cix[i], iy0 = ciy[i], iz0 = ciz[i];

    /* unique neighbor cells (avoid 27× recount when ncell==1) */
    int neigh[27];
    int nneigh = 0;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          int cid = cell_id(ix0 + dx, iy0 + dy, iz0 + dz, nx, ny, nz);
          int seen = 0;
          for (int k = 0; k < nneigh; ++k)
            if (neigh[k] == cid) {
              seen = 1;
              break;
            }
          if (!seen) neigh[nneigh++] = cid;
        }
      }
    }

    for (int k = 0; k < nneigh; ++k) {
      for (int j = head[neigh[k]]; j >= 0; j = next[j]) {
        if (types[j] != tb) continue;
        if (same && j <= i) continue; /* unique i<j */
        if (!same && j == i) continue;

        double dx = xi - pos[3 * j];
        double dy = yi - pos[3 * j + 1];
        double dz = zi - pos[3 * j + 2];
        dx -= Lx * nearbyint(dx * invLx);
        dy -= Ly * nearbyint(dy * invLy);
        dz -= Lz * nearbyint(dz * invLz);
        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= rmax2 || r2 < 1e-24) continue;
        int bin = (int)(sqrt(r2) / dr);
        if (bin >= 0 && bin < nbin) gr->hist[bin] += 1;
      }
    }
  }

  /* unlike pairs: if we only looped centres of type ta, we counted each A-B once.
     If ta!=tb but we used same=false with j any, good.
     Note: for unlike we must NOT require j>i — done. */

  free(head);
  free(next);
  free(cix);
  free(ciy);
  free(ciz);
  gr->nsamples += 1;
  return 0;
}

int staf_gr_normalize(const StafGr *gr, double mean_n_a, double mean_rho_b,
                      int same_type, double *r_out, double *g_out) {
  if (!gr || !r_out || !g_out || mean_n_a <= 0 || mean_rho_b <= 0 ||
      gr->nsamples < 1)
    return -1;
  const double dr = gr->dr;
  for (int i = 0; i < gr->nbin; ++i) {
    const double r_lo = i * dr;
    const double r_hi = (i + 1) * dr;
    const double shell =
        (4.0 / 3.0) * M_PI * (r_hi * r_hi * r_hi - r_lo * r_lo * r_lo);
    r_out[i] = 0.5 * (r_lo + r_hi);
    double denom = (double)gr->nsamples * mean_n_a * mean_rho_b * shell;
    if (same_type) denom *= 0.5; /* hist counted unique pairs */
    g_out[i] =
        (denom > 0.0) ? ((double)gr->hist[i] / denom) : 0.0;
  }
  return 0;
}
