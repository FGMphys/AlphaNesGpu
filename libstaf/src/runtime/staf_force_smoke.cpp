/* One-shot force evaluation smoke: load model_onnx_double + jittered water frame. */
#include "staf.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

static int read_lammps_data(const char* path, std::vector<double>& x,
                            std::vector<int>& type, double box6[6], int& natoms) {
  FILE* f = fopen(path, "r");
  if (!f) return -1;
  char line[1024];
  natoms = 0;
  double xlo = 0, xhi = 0, ylo = 0, yhi = 0, zlo = 0, zhi = 0;
  while (fgets(line, sizeof(line), f)) {
    if (strstr(line, "atoms") && natoms == 0) {
      sscanf(line, "%d", &natoms);
    } else if (strstr(line, "xlo")) {
      sscanf(line, "%lf %lf", &xlo, &xhi);
    } else if (strstr(line, "ylo")) {
      sscanf(line, "%lf %lf", &ylo, &yhi);
    } else if (strstr(line, "zlo")) {
      sscanf(line, "%lf %lf", &zlo, &zhi);
    } else if (strncmp(line, "Atoms", 5) == 0) {
      fgets(line, sizeof(line), f); /* blank */
      x.assign(natoms * 3, 0.0);
      type.assign(natoms, 0);
      for (int i = 0; i < natoms; ++i) {
        int id, t;
        double px, py, pz;
        if (fscanf(f, "%d %d %lf %lf %lf", &id, &t, &px, &py, &pz) != 5)
          return -2;
        type[id - 1] = t - 1;
        x[3 * (id - 1) + 0] = px;
        x[3 * (id - 1) + 1] = py;
        x[3 * (id - 1) + 2] = pz;
      }
      break;
    }
  }
  fclose(f);
  /* jmd box: lx ly lz packed like example — use diagonal lengths */
  box6[0] = xhi - xlo;
  box6[1] = 0;
  box6[2] = 0;
  box6[3] = yhi - ylo;
  box6[4] = 0;
  box6[5] = zhi - zlo;
  return 0;
}

int main(int argc, char** argv) {
  const char* model =
      argc > 1 ? argv[1]
               : "/home/francegm/AlphaNesGpu/test/test-lammps-smoke/model_onnx_double";
  const char* data =
      argc > 2 ? argv[2]
               : "/home/francegm/AlphaNesGpu/test/test-lammps-smoke/data.water_smoke";

  std::vector<double> x;
  std::vector<int> type;
  double box6[6];
  int natoms = 0;
  if (read_lammps_data(data, x, type, box6, natoms) != 0) {
    fprintf(stderr, "failed to read %s\n", data);
    return 1;
  }
  printf("natoms=%d box=(%g,%g,%g)\n", natoms, box6[0], box6[3], box6[5]);

  /* Ensure type.dat + cfg exist in model dir */
  {
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "cp -n /home/francegm/AlphaNesGpu/test/test-lammps-smoke/staf_jmd.cfg %s/ 2>/dev/null; "
             "true",
             model);
    system(cmd);
  }

  StafOptions opt;
  staf_options_default(&opt);
  opt.precision = 1;
  opt.mlp_backend = STAF_MLP_NATIVE;
  opt.device_id = 0;

  StafModel* m = staf_load(model, &opt);
  if (!m) {
    fprintf(stderr, "staf_load failed\n");
    return 2;
  }

  std::vector<double> f(natoms * 3, 0.0), vir(6, 0.0);
  double e = 0.0;
  int rc = staf_compute(m, natoms, 0, x.data(), box6, type.data(), NULL, NULL,
                        0, &e, f.data(), vir.data());
  printf("staf_compute rc=%d  E=%.8f\n", rc, e);
  double fmax = 0.0;
  for (double v : f) fmax = std::fmax(fmax, std::fabs(v));
  printf("|F|_max=%.6g  virial_diag=(%.4g,%.4g,%.4g)\n", fmax, vir[0], vir[1],
         vir[2]);

  staf_free(m);
  if (rc != 0) return 3;
  if (!std::isfinite(e) || !std::isfinite(fmax)) return 4;
  printf("staf_force_smoke: OK\n");
  return 0;
}
