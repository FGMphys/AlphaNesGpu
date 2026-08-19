/* One-shot CG force eval: model_dir + binary frame.
 * Frame (little-endian):
 *   int32 natoms
 *   double box6[6]     (lx, xy, xz, ly, yz, lz)
 *   double xyz[natoms*3]
 * Optional 4th arg: path to write "E\\n fx fy fz\\n..." for Python compare.
 */
#include "staf.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

static bool read_exact(FILE* f, void* p, size_t n) {
  return fread(p, 1, n, f) == n;
}

int main(int argc, char** argv) {
  const char* model =
      argc > 1 ? argv[1]
               : "/home/francegm/AlphaNesGpu/test/test-cg-inference/model_onnx_double";
  const char* frame =
      argc > 2 ? argv[2]
               : "/home/francegm/AlphaNesGpu/test/test-cg-libstaf/frame0.bin";
  const char* out_txt = argc > 3 ? argv[3] : NULL;

  FILE* f = fopen(frame, "rb");
  if (!f) {
    fprintf(stderr, "failed to open %s\n", frame);
    return 1;
  }
  int natoms = 0;
  if (!read_exact(f, &natoms, sizeof(int)) || natoms <= 0) {
    fprintf(stderr, "bad natoms in %s\n", frame);
    fclose(f);
    return 1;
  }
  double box6[6];
  if (!read_exact(f, box6, 6 * sizeof(double))) {
    fclose(f);
    return 1;
  }
  std::vector<double> x((size_t)natoms * 3);
  if (!read_exact(f, x.data(), x.size() * sizeof(double))) {
    fclose(f);
    return 1;
  }
  fclose(f);

  printf("natoms=%d box=(%g,%g,%g)\n", natoms, box6[0], box6[3], box6[5]);

  StafOptions opt;
  staf_options_default(&opt);
  opt.precision = 0; /* float32 ONNX */
  opt.mlp_backend = STAF_MLP_ORT;
  opt.device_id = 0;

  StafModel* m = staf_load(model, &opt);
  if (!m) {
    fprintf(stderr, "staf_load failed\n");
    return 2;
  }

  std::vector<double> force((size_t)natoms * 3, 0.0), vir(6, 0.0);
  double e = 0.0;
  int rc = staf_compute(m, natoms, 0, x.data(), box6, NULL, NULL, NULL, 0, &e,
                        force.data(), vir.data());
  printf("staf_compute rc=%d  E=%.10f\n", rc, e);
  double fmax = 0.0;
  for (double v : force) fmax = std::fmax(fmax, std::fabs(v));
  printf("|F|_max=%.6g  virial_diag=(%.4g,%.4g,%.4g)\n", fmax, vir[0], vir[1],
         vir[2]);

  if (out_txt) {
    FILE* o = fopen(out_txt, "w");
    if (o) {
      fprintf(o, "%.16g\n", e);
      for (int i = 0; i < natoms; ++i)
        fprintf(o, "%.16g %.16g %.16g\n", force[3 * i], force[3 * i + 1],
                force[3 * i + 2]);
      fclose(o);
    }
  }

  staf_free(m);
  if (rc != 0) return 3;
  if (!std::isfinite(e) || !std::isfinite(fmax)) return 4;
  printf("staf_force_smoke: OK\n");
  return 0;
}
