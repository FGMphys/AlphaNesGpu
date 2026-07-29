/* staf_gr.h — multi-species radial distribution g(r) with cell list */
#ifndef STAF_GR_H
#define STAF_GR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct StafGr {
  int nbin;
  double dr;
  double rmax;
  long long *hist; /* length nbin, pair counts */
  int nsamples;
} StafGr;

/* Construct histogram: bins of width dr covering [0, rmax). */
StafGr *staf_gr_create(double dr, double rmax);
void staf_gr_free(StafGr *gr);
void staf_gr_reset(StafGr *gr);

/*
 * Accumulate one frame.
 * pos:   n × 3 (xyz), Å
 * types: n ints (0..ntypes-1)
 * box:   [Lx, Ly, Lz] orthorhombic Å (minimum-image)
 * ta, tb: species pair (ta <= tb for same-type unique pairs; unlike uses all A×B)
 */
int staf_gr_accumulate(StafGr *gr, const double *pos, const int *types, int n,
                       const double *box, int ta, int tb);

/*
 * Normalize into g_out[nbin], r_out[nbin] (bin centers).
 * rho_b = number density of species tb used in ideal-gas shell.
 * n_a   = number of species-ta centres (per frame average if multi-frame:
 *         pass mean n_a and call with total hist / nsamples already in hist,
 *         or pass per-frame n_a and nsamples from gr).
 * For multi-frame: hist already summed; use gr->nsamples, mean_n_a, mean_rho_b.
 */
int staf_gr_normalize(const StafGr *gr, double mean_n_a, double mean_rho_b,
                      int same_type, double *r_out, double *g_out);

#ifdef __cplusplus
}
#endif

#endif
