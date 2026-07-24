#include "staf.h"
#include "staf_mlp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(STAF_WITH_JMD) && STAF_WITH_JMD
extern "C" {
#include "interaction_map.h"
}
#include "nn_nn.h"
#endif

struct StafModel {
  char* model_dir;
  StafOptions opt;
  StafMlp* mlp;
  int n_atoms;
  int initialized;
#if defined(STAF_WITH_JMD) && STAF_WITH_JMD
  interactionmap* ime;
  vector* pos;
  vector* force;
  vector virial_diag;
  int* order_to_jmd; /* LAMMPS local i -> JMD slot (type-sorted) */
  int* jmd_to_order; /* JMD slot -> LAMMPS local i */
#endif
};

void staf_options_default(StafOptions* opt) {
  if (!opt) return;
  memset(opt, 0, sizeof(*opt));
  opt->mlp_backend = STAF_MLP_ORT;
  opt->device_id = 0;
  opt->precision = 0; /* float32 — ORT grad export (FP64 ORT training broken) */
}

StafModel* staf_load(const char* model_dir, const StafOptions* opt) {
  if (!model_dir) return NULL;
  StafModel* m = (StafModel*)calloc(1, sizeof(StafModel));
  if (!m) return NULL;
  staf_options_default(&m->opt);
  if (opt) m->opt = *opt;
  m->model_dir = (char*)malloc(strlen(model_dir) + 1);
  if (!m->model_dir) {
    free(m);
    return NULL;
  }
  memcpy(m->model_dir, model_dir, strlen(model_dir) + 1);

  m->mlp = staf_mlp_create(m->opt.mlp_backend, model_dir, m->opt.precision,
                           m->opt.device_id);
  if (!m->mlp) {
    fprintf(stderr, "staf_load: staf_mlp_create failed for %s\n", model_dir);
    staf_free(m);
    return NULL;
  }

#if defined(STAF_WITH_JMD) && STAF_WITH_JMD
  staf_jmd_set_mlp(m->mlp);
#else
  fprintf(stderr,
          "staf_load: built without STAF_WITH_JMD; "
          "staf_compute will not run CUDA AF/force\n");
#endif
  return m;
}

int staf_compute(StafModel* m, int nlocal, int nghost, const double* x,
                 const double* box, const int* type, double* e_rank, double* f,
                 double* virial) {
  (void)type;
  if (!m || !m->mlp) return -1;

#if !(defined(STAF_WITH_JMD) && STAF_WITH_JMD)
  (void)nlocal;
  (void)nghost;
  (void)x;
  (void)box;
  if (e_rank) *e_rank = 0.0;
  if (virial)
    for (int i = 0; i < 6; ++i) virial[i] = 0.0;
  return -2;
#else
  /* Smoke / single-rank: use nlocal only (ignore ghosts for MVP). */
  int N = nlocal;
  if (N <= 0) return -3;

  if (!m->initialized) {
    char cfg_path[4096];
    snprintf(cfg_path, sizeof(cfg_path), "%s/staf_jmd.cfg", m->model_dir);
    FILE* cfg = fopen(cfg_path, "r");
    if (!cfg) {
      /* synthesize minimal config in model_dir */
      snprintf(cfg_path, sizeof(cfg_path), "%s/.staf_jmd_auto.cfg", m->model_dir);
      cfg = fopen(cfg_path, "w");
      if (!cfg) return -4;
      fprintf(cfg,
              "Buffer_radial_descriptors = 60\n"
              "Buffer_angular_descriptors = 1770\n"
              "Cutoff = 4.5\n"
              "Cutoff_Angular = 4.5\n"
              "Hardcore_Cutoff = 2.25\n"
              "number_of_types = 2\n"
              "file_of_tipos = %s/type.dat\n"
              "nn_export_dir = %s\n",
              m->model_dir, m->model_dir);
      fclose(cfg);
      cfg = fopen(cfg_path, "r");
    }
    if (!cfg) return -4;
    /* chdir into model_dir so relative type.dat / alpha paths resolve */
    char cwd[4096];
    if (!getcwd(cwd, sizeof(cwd))) return -5;
    if (chdir(m->model_dir) != 0) {
      fclose(cfg);
      return -5;
    }
    FILE* cfg2 = fopen("staf_jmd.cfg", "r");
    if (!cfg2) cfg2 = fopen(".staf_jmd_auto.cfg", "r");
    if (!cfg2) {
      /* rewrite with relative paths */
      cfg2 = fopen(".staf_jmd_auto.cfg", "w");
      fprintf(cfg2,
              "Buffer_radial_descriptors = 60\n"
              "Buffer_angular_descriptors = 1770\n"
              "Cutoff = 4.5\n"
              "Cutoff_Angular = 4.5\n"
              "Hardcore_Cutoff = 2.25\n"
              "number_of_types = 2\n"
              "file_of_tipos = type.dat\n"
              "nn_export_dir = .\n");
      fclose(cfg2);
      cfg2 = fopen(".staf_jmd_auto.cfg", "r");
    }
    fclose(cfg);
    initializenn_(cfg2, N);
    fclose(cfg2);
    chdir(cwd);

    m->ime = createInteractionMap(N, 60);
    m->pos = (vector*)calloc(N, sizeof(vector));
    m->force = (vector*)calloc(N, sizeof(vector));
    m->order_to_jmd = (int*)calloc(N, sizeof(int));
    m->jmd_to_order = (int*)calloc(N, sizeof(int));
    m->n_atoms = N;
    m->initialized = 1;
  }

  if (N != m->n_atoms) {
    fprintf(stderr, "staf_compute: N changed (%d -> %d); not supported\n",
            m->n_atoms, N);
    return -6;
  }

  double box6[6];
  /* LAMMPS / STAF triclinic packing: [xx, xy, xz, yy, yz, zz] */
  box6[0] = box[0];
  box6[1] = box[1];
  box6[2] = box[2];
  box6[3] = box[3];
  box6[4] = box[4];
  box6[5] = box[5];

  /*
   * neuralmdGPU / JMD expects:
   *  1) positions in fractional (scaled) coordinates
   *  2) atoms packed contiguously by species (type.dat counts), because
   *     Des2b/AF buffers are split as [type0 block | type1 block | ...]
   * LAMMPS may spatially reorder atoms — pack by `type` here and scatter
   * forces back to the caller's order.
   */
  const double bx0 = box6[0], bx1 = box6[1], bx2 = box6[2];
  const double bx3 = box6[3], bx4 = box6[4], bx5 = box6[5];
  if (bx0 == 0.0 || bx3 == 0.0 || bx5 == 0.0) return -7;
  const double ino0 = 1.0 / bx0;
  const double ino1 = -bx1 / (bx0 * bx3);
  const double ino2 =
      (bx1 * bx4) / (bx0 * bx3 * bx5) - bx2 / (bx0 * bx5);
  const double ino3 = 1.0 / bx3;
  const double ino4 = -bx4 / (bx3 * bx5);
  const double ino5 = 1.0 / bx5;

  /* Build type-sorted permutation (stable within type by ascending local i). */
  if (type) {
    int n_type0 = 0, n_type1 = 0;
    for (int i = 0; i < N; ++i) {
      if (type[i] == 0) n_type0++;
      else if (type[i] == 1) n_type1++;
      else {
        fprintf(stderr, "staf_compute: unsupported type %d at atom %d\n",
                type[i], i);
        return -8;
      }
    }
    if (n_type0 + n_type1 != N) return -8;
    int c0 = 0, c1 = n_type0;
    for (int i = 0; i < N; ++i) {
      int slot = (type[i] == 0) ? c0++ : c1++;
      m->order_to_jmd[i] = slot;
      m->jmd_to_order[slot] = i;
    }
  } else {
    for (int i = 0; i < N; ++i) {
      m->order_to_jmd[i] = i;
      m->jmd_to_order[i] = i;
    }
  }

  for (int i = 0; i < N; ++i) {
    const double cx = x[3 * i];
    const double cy = x[3 * i + 1];
    const double cz = x[3 * i + 2];
    const int j = m->order_to_jmd[i];
    m->pos[j].x = ino0 * cx + ino1 * cy + ino2 * cz;
    m->pos[j].y = ino3 * cy + ino4 * cz;
    m->pos[j].z = ino5 * cz;
  }

  {
    const char *pp = getenv("STAF_DUMP_POS");
    if (pp && pp[0]) {
      FILE *df = fopen(pp, "wb");
      if (df) {
        fwrite(&N, sizeof(int), 1, df);
        fwrite(box6, sizeof(double), 6, df);
        for (int j = 0; j < N; ++j) {
          int i = m->jmd_to_order[j];
          double cart[3] = {x[3 * i], x[3 * i + 1], x[3 * i + 2]};
          double frac[3] = {m->pos[j].x, m->pos[j].y, m->pos[j].z};
          fwrite(cart, sizeof(double), 3, df);
          fwrite(frac, sizeof(double), 3, df);
        }
        fclose(df);
        fprintf(stderr, "staf_compute: dumped cart/frac pos (JMD order) to %s\n",
                pp);
      }
    }
  }

  double energy = 0.0, vir = 0.0;
  calculateforces(m->pos, box6, m->ime, &energy, m->force, &vir,
                  &m->virial_diag);

  if (e_rank) *e_rank = energy;
  if (f) {
    for (int j = 0; j < N; ++j) {
      int i = m->jmd_to_order[j];
      f[3 * i] = m->force[j].x;
      f[3 * i + 1] = m->force[j].y;
      f[3 * i + 2] = m->force[j].z;
    }
    for (int i = N; i < N + nghost; ++i) {
      f[3 * i] = f[3 * i + 1] = f[3 * i + 2] = 0.0;
    }
  }
  if (virial) {
    virial[0] = m->virial_diag.x;
    virial[1] = m->virial_diag.y;
    virial[2] = m->virial_diag.z;
    virial[3] = virial[4] = virial[5] = 0.0;
  }
  return 0;
#endif
}

void staf_free(StafModel* m) {
  if (!m) return;
#if defined(STAF_WITH_JMD) && STAF_WITH_JMD
  if (m->initialized) {
    nnDestructor();
    /* skip freeInteractionMap — allocator mismatch can abort on smoke exit */
    free(m->pos);
    free(m->force);
    free(m->order_to_jmd);
    free(m->jmd_to_order);
  }
#endif
  if (m->mlp) staf_mlp_destroy(m->mlp);
  free(m->model_dir);
  free(m);
}
