#include "staf.h"
#include "staf_mlp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#if defined(STAF_WITH_JMD) && STAF_WITH_JMD
#include <cuda_runtime.h>
#endif

#if defined(STAF_WITH_JMD) && STAF_WITH_JMD
#include "vector.h"
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
  int cap_nall;
  int initialized;
#if defined(STAF_WITH_JMD) && STAF_WITH_JMD
  interactionmap* ime;
  vector* pos;
  vector* force;
  vector virial_diag;
  int* order_to_jmd;
  int* jmd_to_order;
  int num_types;
#endif
};

void staf_options_default(StafOptions* opt) {
  if (!opt) return;
  memset(opt, 0, sizeof(*opt));
  opt->mlp_backend = STAF_MLP_ORT;
  opt->device_id = 0;
  opt->precision = 0; /* float32 — ORT grad export (FP64 ORT training broken) */
}

int staf_cuda_set_device(int device_id) {
#if defined(STAF_WITH_JMD) && STAF_WITH_JMD
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) {
    fprintf(stderr, "staf_cuda_set_device: no CUDA devices\n");
    return -1;
  }
  if (device_id < 0) device_id = 0;
  device_id = device_id % ndev;
  cudaError_t err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    fprintf(stderr, "staf_cuda_set_device(%d): %s\n", device_id,
            cudaGetErrorString(err));
    return -1;
  }
  return device_id;
#else
  (void)device_id;
  return 0;
#endif
}

StafModel* staf_load(const char* model_dir, const StafOptions* opt) {
  if (!model_dir) return NULL;
  StafModel* m = (StafModel*)calloc(1, sizeof(StafModel));
  if (!m) return NULL;
  staf_options_default(&m->opt);
  if (opt) m->opt = *opt;
  {
    int dev = staf_cuda_set_device(m->opt.device_id);
    if (dev < 0) {
      staf_free(m);
      return NULL;
    }
    m->opt.device_id = dev;
  }
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

#if defined(STAF_WITH_JMD) && STAF_WITH_JMD
static int staf_ensure_buffers(StafModel* m, int nall) {
  if (nall <= m->cap_nall) return 0;
  free(m->pos);
  free(m->force);
  free(m->order_to_jmd);
  free(m->jmd_to_order);
  m->pos = (vector*)calloc((size_t)nall, sizeof(vector));
  m->force = (vector*)calloc((size_t)nall, sizeof(vector));
  m->order_to_jmd = (int*)calloc((size_t)nall, sizeof(int));
  m->jmd_to_order = (int*)calloc((size_t)nall, sizeof(int));
  if (!m->pos || !m->force || !m->order_to_jmd || !m->jmd_to_order)
    return -1;
  if (m->ime) m->ime = NULL;
  m->ime = createInteractionMap(nall, staf_jmd_radial_buffer());
  if (!m->ime) return -1;
  m->cap_nall = nall;
  return 0;
}

static int read_cutoff_info(const char* dir, double* rc, int* rad_buff,
                            double* rc_ang, int* ang_buff, double* rs,
                            double* rc_inter, double* ra_inter,
                            double* rs_inter) {
  /* 1-epoch origami defaults (input_epoch1.yaml). */
  *rc = 50.0;
  *rad_buff = 24;
  *rc_ang = 50.0;
  *ang_buff = 276;
  *rs = 25.0;
  *rc_inter = 20.0;
  *ra_inter = 20.0;
  *rs_inter = 10.0;
  char path[4096];
  snprintf(path, sizeof(path), "%s/cutoff_info", dir);
  FILE* f = fopen(path, "r");
  if (!f) return 0;
  double dummy;
  if (fscanf(f, "%lf %d", rc, rad_buff) != 2) {
    fclose(f);
    return -1;
  }
  if (fscanf(f, "%lf %d", rc_ang, ang_buff) != 2) {
    fclose(f);
    return -1;
  }
  if (fscanf(f, "%lf %lf", rs, &dummy) != 2) {
    fclose(f);
    return -1;
  }
  if (fscanf(f, "%lf %lf", rc_inter, &dummy) != 2) {
    fclose(f);
    return -1;
  }
  if (fscanf(f, "%lf %lf", ra_inter, &dummy) != 2) {
    fclose(f);
    return -1;
  }
  if (fscanf(f, "%lf %lf", rs_inter, &dummy) != 2) {
    fclose(f);
    return -1;
  }
  fclose(f);
  return 1;
}

static int read_number_of_nn(const char* dir) {
  char path[4096];
  snprintf(path, sizeof(path), "%s/number_of_nn.dat", dir);
  FILE* f = fopen(path, "r");
  if (!f) return 1;
  int n = 1;
  if (fscanf(f, "%d", &n) != 1) n = 1;
  fclose(f);
  return n < 1 ? 1 : n;
}

static int staf_init_jmd(StafModel* m, int nlocal) {
  double rc, rc_ang, rs, rc_inter, ra_inter, rs_inter;
  int rad_buff, ang_buff;
  read_cutoff_info(m->model_dir, &rc, &rad_buff, &rc_ang, &ang_buff, &rs,
                   &rc_inter, &ra_inter, &rs_inter);
  int n_nn = read_number_of_nn(m->model_dir);

  char cfg_path[4096];
  snprintf(cfg_path, sizeof(cfg_path), "%s/staf_jmd.cfg", m->model_dir);
  FILE* cfg = fopen(cfg_path, "r");
  if (!cfg) {
    snprintf(cfg_path, sizeof(cfg_path), "%s/.staf_jmd_auto.cfg", m->model_dir);
    cfg = fopen(cfg_path, "w");
    if (!cfg) return -4;
    fprintf(cfg,
            "Buffer_radial_descriptors = %d\n"
            "Buffer_angular_descriptors = %d\n"
            "Cutoff = %g\n"
            "Cutoff_Angular = %g\n"
            "Hardcore_Cutoff = %g\n"
            "Cutoff_Inter = %g\n"
            "Hardcore_Cutoff_Inter = %g\n"
            "Cutoff_Angular_Inter = %g\n"
            "number_of_NN = %d\n"
            "file_map_intra = map_intra.dat\n"
            "file_color_type_map = color_type_map.dat\n"
            "file_map_color_interaction = map_color_interaction.dat\n"
            "nn_export_dir = .\n",
            rad_buff, ang_buff, rc, rc_ang, rs, rc_inter, rs_inter, ra_inter,
            n_nn);
    fclose(cfg);
  } else {
    fclose(cfg);
  }

  char cwd[4096];
  if (!getcwd(cwd, sizeof(cwd))) return -5;
  if (chdir(m->model_dir) != 0) return -5;

  FILE* cfg2 = fopen("staf_jmd.cfg", "r");
  if (!cfg2) cfg2 = fopen(".staf_jmd_auto.cfg", "r");
  if (!cfg2) {
    if (chdir(cwd) != 0) return -5;
    return -4;
  }
  initializenn_(cfg2, nlocal);
  fclose(cfg2);
  if (chdir(cwd) != 0) return -5;

  m->num_types = staf_jmd_num_types();
  if (staf_ensure_buffers(m, nlocal) != 0) return -1;
  m->n_atoms = nlocal;
  m->initialized = 1;
  return 0;
}
#endif

int staf_compute(StafModel* m, int nlocal, int nghost, const double* x,
                 const double* box, const int* type, const int* howmany,
                 const int* with, int maxneigh, double* e_rank, double* f,
                 double* virial) {
  if (!m || !m->mlp) return -1;

#if !(defined(STAF_WITH_JMD) && STAF_WITH_JMD)
  (void)nlocal;
  (void)nghost;
  (void)x;
  (void)box;
  (void)type;
  (void)howmany;
  (void)with;
  (void)maxneigh;
  if (e_rank) *e_rank = 0.0;
  if (virial)
    for (int i = 0; i < 6; ++i) virial[i] = 0.0;
  return -2;
#else
  const int use_ext_neigh = (howmany != NULL);
  const int nall = use_ext_neigh ? (nlocal + nghost) : nlocal;
  if (nlocal <= 0) return -3;

  if (!m->initialized) {
    int rc = staf_init_jmd(m, nlocal);
    if (rc != 0) return rc;
  }

  if (staf_ensure_buffers(m, nall) != 0) return -1;

  double box6[6];
  box6[0] = box[0];
  box6[1] = box[1];
  box6[2] = box[2];
  box6[3] = box[3];
  box6[4] = box[4];
  box6[5] = box[5];
  if (box6[0] == 0.0 || box6[3] == 0.0 || box6[5] == 0.0) return -7;

  const double ino0 = 1.0 / box6[0];
  const double ino1 = -box6[1] / (box6[0] * box6[3]);
  const double ino2 =
      (box6[1] * box6[4]) / (box6[0] * box6[3] * box6[5]) - box6[2] / (box6[0] * box6[5]);
  const double ino3 = 1.0 / box6[3];
  const double ino4 = -box6[4] / (box6[3] * box6[5]);
  const double ino5 = 1.0 / box6[5];

  /* type[] is 0-based bead id (LAMMPS tag-1) so ghosts inherit color/map_intra. */
  int tipos_owned[1] = {nlocal};
  int* type_map_all = (int*)calloc((size_t)nall, sizeof(int));
  if (!type_map_all) return -1;
  for (int i = 0; i < nall; ++i) {
    m->order_to_jmd[i] = i;
    m->jmd_to_order[i] = i;
    type_map_all[i] = type ? type[i] : i;
  }

  if (staf_jmd_resize(nlocal, nall, tipos_owned, type_map_all) != 0) {
    free(type_map_all);
    return -1;
  }

  if (use_ext_neigh) {
    const int rb = staf_jmd_radial_buffer();
    int* how_jmd = (int*)calloc((size_t)nlocal, sizeof(int));
    int* with_jmd = (int*)calloc((size_t)nlocal * (size_t)rb, sizeof(int));
    if (!how_jmd || !with_jmd) {
      free(how_jmd);
      free(with_jmd);
      free(type_map_all);
      return -1;
    }
    for (int i = 0; i < nlocal; ++i) {
      int nn = howmany[i];
      if (nn < 0) nn = 0;
      if (nn > rb) nn = rb;
      if (nn > maxneigh) nn = maxneigh;
      how_jmd[i] = nn;
      for (int k = 0; k < nn; ++k) {
        const int j_lmp = with[i * maxneigh + k];
        if (j_lmp < 0 || j_lmp >= nall) {
          free(how_jmd);
          free(with_jmd);
          free(type_map_all);
          return -9;
        }
        with_jmd[i * rb + k] = j_lmp;
      }
    }
    staf_jmd_set_external_neigh(how_jmd, with_jmd, nlocal, rb);
    free(how_jmd);
    free(with_jmd);
  } else {
    staf_jmd_clear_external_neigh();
  }

  for (int i = 0; i < nall; ++i) {
    const double cx = x[3 * i];
    const double cy = x[3 * i + 1];
    const double cz = x[3 * i + 2];
    m->pos[i].x = ino0 * cx + ino1 * cy + ino2 * cz;
    m->pos[i].y = ino3 * cy + ino4 * cz;
    m->pos[i].z = ino5 * cz;
  }

  double energy = 0.0, vir = 0.0;
  calculateforces(m->pos, box6, m->ime, &energy, m->force, &vir,
                  &m->virial_diag);

  if (!isfinite(energy)) {
    fprintf(stderr, "staf_compute: non-finite energy=%g nlocal=%d nghost=%d\n",
            energy, nlocal, nghost);
  }

  if (e_rank) *e_rank = energy;
  if (f) {
    for (int i = 0; i < nall; ++i) {
      f[3 * i] = m->force[i].x;
      f[3 * i + 1] = m->force[i].y;
      f[3 * i + 2] = m->force[i].z;
    }
  }
  if (virial) {
    virial[0] = m->virial_diag.x;
    virial[1] = m->virial_diag.y;
    virial[2] = m->virial_diag.z;
    virial[3] = virial[4] = virial[5] = 0.0;
  }

  m->n_atoms = nall;
  free(type_map_all);
  return 0;
#endif
}

void staf_free(StafModel* m) {
  if (!m) return;
#if defined(STAF_WITH_JMD) && STAF_WITH_JMD
  if (m->initialized) {
    nnDestructor();
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
