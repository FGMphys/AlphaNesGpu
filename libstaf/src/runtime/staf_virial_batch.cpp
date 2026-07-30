/* staf_virial_batch: load ONNX model once, eval virial diag on many frames.
 * Binary input (little-endian):
 *   int32 nframes, natoms
 *   int32 types[natoms]   (0-based)
 *   per frame:
 *     double box6[6]      (Lx,0,0, Ly,0, Lz) jmd packing
 *     double xyz[natoms*3]
 * stdout: one line per frame: E  Wxx Wyy Wzz
 */
#include "staf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

static bool read_exact(FILE* f, void* p, size_t n) {
  return fread(p, 1, n, f) == n;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s model_dir frames.bin [float|double]\n", argv[0]);
    return 1;
  }
  const char* model = argv[1];
  const char* binpath = argv[2];
  int prec = 0;
  if (argc >= 4 && strcmp(argv[3], "double") == 0) prec = 1;

  FILE* f = fopen(binpath, "rb");
  if (!f) {
    perror(binpath);
    return 2;
  }
  int32_t nframes = 0, natoms = 0;
  if (!read_exact(f, &nframes, 4) || !read_exact(f, &natoms, 4) || nframes < 1 ||
      natoms < 1) {
    fprintf(stderr, "bad header\n");
    return 3;
  }
  std::vector<int> type((size_t)natoms);
  if (!read_exact(f, type.data(), sizeof(int) * (size_t)natoms)) {
    fprintf(stderr, "bad types\n");
    return 4;
  }

  StafOptions opt;
  staf_options_default(&opt);
  opt.precision = prec;
  opt.mlp_backend = STAF_MLP_ORT;
  opt.device_id = 0;
  StafModel* m = staf_load(model, &opt);
  if (!m) {
    fprintf(stderr, "staf_load failed: %s\n", model);
    return 5;
  }

  std::vector<double> x((size_t)natoms * 3), force((size_t)natoms * 3), vir(6);
  double box6[6];
  for (int fi = 0; fi < nframes; ++fi) {
    if (!read_exact(f, box6, sizeof(box6)) ||
        !read_exact(f, x.data(), sizeof(double) * (size_t)natoms * 3)) {
      fprintf(stderr, "bad frame %d\n", fi);
      staf_free(m);
      return 6;
    }
    std::fill(force.begin(), force.end(), 0.0);
    std::fill(vir.begin(), vir.end(), 0.0);
    double e = 0.0;
    int rc = staf_compute(m, natoms, 0, x.data(), box6, type.data(), NULL, NULL,
                          0, &e, force.data(), vir.data());
    if (rc != 0 || !std::isfinite(e) || !std::isfinite(vir[0])) {
      printf("nan nan nan nan\n");
      continue;
    }
    printf("%.10g %.10g %.10g %.10g\n", e, vir[0], vir[1], vir[2]);
    fflush(stdout);
  }
  staf_free(m);
  fclose(f);
  return 0;
}
