/*
 * nn_nn_mlp.cu
 *
 * Patched from neuralmdGPU/full_atom/src/nn_nn.cu:
 *   - Removed all #include <tensorflow/...>
 *   - Removed nnmodel / TF session management
 *   - Constructor_TensorFlow_Model → Constructor_MLP_Model
 *     allocates Gradients / Gradients_d only; no TF session loaded.
 *   - Compute_NNEnergyandGradient_all: packs AFs from device → host,
 *     calls staf_mlp_eval (precision=1, double), scatters gradients back.
 *   - Compute_NNEnergyandGradient(type) kept as thin wrapper used only
 *     by Compute_NNEnergyandGradient_all (direct per-type scatter).
 *   - deletetensor / nnDestructor: no-ops (no TF tensors to delete).
 *   - staf_jmd_set_mlp: stores the external StafMlp* in a static global.
 *
 * Compile:
 *   nvcc -dc -O2 -std=c++17 \
 *        -I<vendor/jmd> -I<libstaf/include> \
 *        nn_nn_mlp.cu -o nn_nn_mlp.o
 *
 * Link with: nn_nn_mlp.o  <all other neuralmdGPU .o except nn_nn.o>
 *            -lstaf -lcuda -lcudart
 * (see README.md for the full .o list)
 *
 * GSL: vector.h declares randomVector(gsl_rng*).  Define JMD_NO_GSL if
 * you do not link GSL — randomVector will not be declared/available.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>

/* GSL is used by vector.c (randomVector).  Include it here so that the
 * compiler sees gsl_rng before vector.h declares randomVector. */
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <cuda.h>
#include <cuda_runtime.h>

/* libstaf MLP API  (compile with -I libstaf/include) */
#include "staf_mlp.h"

/* vendor/jmd local headers */
#include "vector.h"
#include "interaction_map.h"
extern "C" {
#include "secure_search.h"
#include "nn_io.h"
#include "nn_smart_allocator.h"
}
#include "nn_smart_allocator_gpu.h"
#include "global_definitions.h"
#include "io.h"
#include "nn_nn.h"
#include "celle_gpu.h"
#include "src_nn/descriptor_builder/reforce.h"
#include "src_nn/fingerprint/rad/reforce.h"
#include "src_nn/fingerprint/ang/reforce.h"
#include "src_nn/force/rad/reforce.h"
#include "src_nn/force/ang/reforce.h"

/* ===================================================================== */
/*  Utility macros                                                        */
/* ===================================================================== */
#define SQR(x)       ((x) * (x))
#define Sqrt(x)      (sqrt(x))
#define Power(x, n)  (pow(x, n))

/* ===================================================================== */
/*  Module-level static state (mirrors original nn_nn.cu)                */
/* ===================================================================== */

/* Descriptor geometry */
static int    N;
static int    Radial_Buffer;
static int    Angular_Buffer;
static double Cutoff;
static double Cutoff_Angular;

/* GPU buffers – cell / neighbour list */
static double *box_d, *inobox_d, *pos_d;
static int    *howmany_d, *with_d;
static double *with_dist2_d;
static int    *Cells;
static int    *Cells_howmany;
static int     MAX_PARTICLE_CELLS;

/* Descriptor device arrays */
static int    **Intmap2b_d;
static int    **Intmap3b_d;
static double **Des2b_d;
static double **Des3b_d;
static double **Des3bsupp_d;
static double **Der2b_d;
static double **Der3bsupp_d;
static double **Der3b_d;
static int    **Numtriplet_d;

/* Type bookkeeping */
static int  *Tipos;
static int  *Tipos_d;
static int  *Type_map;
static int  *Type_map_d;

/* Atomic-fingerprint (AF) arrays */
static double **Alpha;
static double **Alpha_a;
static double **Alpha_d;
static double **Alpha_a_d;
static int      dimAFSall_tot;
static int     *Alpha_num;
static int     *Alpha_a_num;
static int     *Alpha_num_d;
static int     *Alpha_a_num_d;

static double **Coeff_2b;
static double **Coeff_3b;
static double **Coeff_2b_d;
static double **Coeff_3b_d;

/* Packed AF device buffer [NumTypes][Tipos[t]*(Alpha_num[t]+Alpha_a_num[t])] */
static double **AFs_all_d;
static int      NumTypes;
static int      NumTypesCouples;
static char     NNmodelRoot[1000];
static char     Typefile[1000];

/* Gradient buffers (host + device) */
static double **Gradients;
static double **Gradients_d;

/* Force / virial */
static double *Force_d;
static double *Virial_d;
static double *Virial_Diagonal_d;

/* Repulsion parameters */
static double Rs, coeffA, coeffB, coeffC, Pow_alpha, Pow_beta;

/* ===================================================================== */
/*  StafMlp handle set externally via staf_jmd_set_mlp()                 */
/* ===================================================================== */
static StafMlp *g_staf_mlp = NULL;

extern "C" void staf_jmd_set_mlp(void *mlp)
{
    g_staf_mlp = (StafMlp *)mlp;
}

/* ===================================================================== */
/*  Repulsion construction (unchanged)                                    */
/* ===================================================================== */

static void save_cutoff(double rc)
{
    FILE *newfile = fopen("cutoff_curve.dat", "w");
    double dx = rc / 1000.;
    double pi = 3.1415926535;
    double x = 0;
    for (int k = 0; k < 1000; k++) {
        x = x + dx;
        if (x < Rs)
            fprintf(newfile, "%g %g\n", x,
                    coeffA / Power(x, Pow_alpha) +
                    coeffB / Power(x, Pow_beta) + coeffC);
        else
            fprintf(newfile, "%g %g\n", x, 0.5 * (1 + cos(pi * x / rc)));
    }
    fclose(newfile);
}

static void construct_repulsion(double rc)
{
    double alpha = 1.;
    double beta  = -30.;
    double pi    = 3.1415926535;
    Pow_alpha = alpha;
    Pow_beta  = beta;
    double rs = Rs;
    double f  = 0.5 * (cos(pi * rs / rc) + 1);
    double f1 = -0.5 * pi / rc * sin(pi * rs / rc);
    double f2_red    = -0.5 * SQR(pi / rc) * cos(pi * rs / rc) * SQR(rs);
    double gamma_red = 1. / (alpha - beta) * alpha - 1;
    double delta_red = 1. / (alpha - beta) * (f * (alpha - beta) - f1 * rs - f * alpha);
    double eta_red   = -alpha / (alpha - beta);
    double epsilon_red = 1. / (alpha - beta) * (rs * f1 + alpha * f);
    double c2_red = alpha * (alpha + 1) * delta_red + beta * (beta + 1) * epsilon_red;
    double c1_red = alpha * (alpha + 1) * gamma_red + beta * (beta + 1) * eta_red;
    coeffC = (f2_red - c2_red) / c1_red;
    double eta     = -alpha * Power(rs, beta) / (alpha - beta);
    double epsilon = Power(rs, beta) / (alpha - beta) * (rs * f1 + alpha * f);
    coeffB = eta * coeffC + epsilon;
    double gamma = Power(rs, alpha) / (alpha - beta) * alpha - Power(rs, alpha);
    double delta = Power(rs, alpha) / (alpha - beta) * (f * (alpha - beta) - f1 * rs - f * alpha);
    coeffA = gamma * coeffC + delta;
    save_cutoff(rc);
}

/* ===================================================================== */
/*  Constructor_Descriptors  (unchanged from original)                    */
/* ===================================================================== */
void Constructor_Descriptors(FILE *config_file)
{
    SearchTable *s = searchNew();
    searchInt("Buffer_angular_descriptors", &Angular_Buffer, s);
    searchDouble("Cutoff",          &Cutoff, s);
    searchDouble("Hardcore_Cutoff", &Rs, s);
    searchDouble("Cutoff_Angular",  &Cutoff_Angular, s);
    searchString("nn_export_dir",   NNmodelRoot, s);
    searchString("file_of_tipos",   Typefile, s);
    searchInt("number_of_types",    &NumTypes, s);
    searchInt("Buffer_radial_descriptors", &Radial_Buffer, s);
    searchFile(config_file, s);
    searchFree(s);

    construct_repulsion(Cutoff);

    init_block_dim(Radial_Buffer);
    init_block_dim_ang(Angular_Buffer);

    NumTypesCouples = NumTypes + NumTypes * (NumTypes - 1) / 2;

    Tipos = (int *)calloc(NumTypes, sizeof(int));
    cudaMalloc((void **)&Tipos_d, NumTypes * sizeof(int));
    read_tipos(Tipos, NumTypes, Typefile);
    cudaMemcpy(Tipos_d, Tipos, NumTypes * sizeof(int), cudaMemcpyHostToDevice);

    Type_map = (int *)calloc(N, sizeof(int));
    cudaMalloc((void **)&Type_map_d, N * sizeof(int));
    make_typemap(Type_map, N, NNmodelRoot);
    cudaMemcpy(Type_map_d, Type_map, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&box_d,     6 * sizeof(double));
    cudaMalloc((void **)&inobox_d,  6 * sizeof(double));
    cudaMalloc((void **)&pos_d,     3 * N * sizeof(double));
    cudaMalloc((void **)&howmany_d, N * sizeof(int));
    cudaMalloc((void **)&with_d,    N * Radial_Buffer * sizeof(int));

    MAX_PARTICLE_CELLS = Radial_Buffer;
    cudaMalloc(&with_dist2_d, N * Radial_Buffer * sizeof(double));

    Intmap2b_d  = createIrregularMatrix2D_CUDA_int(NumTypes, Tipos, (Radial_Buffer + 1));
    Intmap3b_d  = createIrregularMatrix2D_CUDA_int(NumTypes, Tipos, (Angular_Buffer * 2));
    Des2b_d     = createIrregularMatrix2D_CUDA(NumTypes, Tipos, Radial_Buffer);
    Des3b_d     = createIrregularMatrix2D_CUDA(NumTypes, Tipos, Angular_Buffer);
    Des3bsupp_d = createIrregularMatrix2D_CUDA(NumTypes, Tipos, Radial_Buffer);
    Der2b_d     = createIrregularMatrix2D_CUDA(NumTypes, Tipos, (Radial_Buffer * 3));
    Der3bsupp_d = createIrregularMatrix2D_CUDA(NumTypes, Tipos, (Radial_Buffer * 3));
    Der3b_d     = createIrregularMatrix2D_CUDA(NumTypes, Tipos, (Angular_Buffer * 2 * 3));
    Numtriplet_d = createIrregularMatrix2D_CUDA_int(NumTypes, Tipos, 1);

    cudaMalloc((void **)&Force_d,            3 * N * sizeof(double));
    cudaMalloc((void **)&Virial_d,           sizeof(double));
    cudaMalloc((void **)&Virial_Diagonal_d,  3 * sizeof(double));
}

/* ===================================================================== */
/*  make_typemap  (unchanged)                                             */
/* ===================================================================== */
void make_typemap(int *type_map, int num_of_particles, char *root_path)
{
    int code = 0, par = 0;
    for (int k = 0; k < NumTypes; k++) {
        for (int y = 0; y < Tipos[k]; y++) {
            type_map[par] = code;
            par++;
        }
        code++;
    }
    assert(N == par);
    (void)root_path;
}

/* ===================================================================== */
/*  Constructor_AFS  (unchanged)                                          */
/* ===================================================================== */
void Constructor_AFS()
{
    Alpha_num   = (int *)calloc(NumTypes, sizeof(int));
    Alpha_a_num = (int *)calloc(NumTypes, sizeof(int));
    cudaMalloc((void **)&Alpha_num_d,   NumTypes * sizeof(int));
    cudaMalloc((void **)&Alpha_a_num_d, NumTypes * sizeof(int));

    readAlpha_num(NNmodelRoot, Alpha_num, Alpha_a_num, NumTypes);
    print_data_int(Alpha_num,   NumTypes);
    print_data_int(Alpha_a_num, NumTypes);

    cudaMemcpy(Alpha_num_d,   Alpha_num,   NumTypes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Alpha_a_num_d, Alpha_a_num, NumTypes * sizeof(int), cudaMemcpyHostToDevice);

    Alpha    = IrregularMatrix2Ddouble(NumTypes, Alpha_num,   NumTypes);
    Alpha_a  = IrregularMatrix2Ddouble(NumTypes, Alpha_a_num, 3 * NumTypesCouples);
    Coeff_2b = IrregularMatrix2Ddouble(NumTypes, Alpha_num,   NumTypes);
    Coeff_3b = IrregularMatrix2Ddouble(NumTypes, Alpha_a_num, NumTypesCouples);

    Alpha_d    = createIrregularMatrix2D_CUDA(NumTypes, Alpha_num,   NumTypes);
    Alpha_a_d  = createIrregularMatrix2D_CUDA(NumTypes, Alpha_a_num, 3 * NumTypesCouples);
    Coeff_2b_d = createIrregularMatrix2D_CUDA(NumTypes, Alpha_num,   NumTypes);
    Coeff_3b_d = createIrregularMatrix2D_CUDA(NumTypes, Alpha_a_num, NumTypesCouples);

    for (int type = 0; type < NumTypes; type++) {
        readalpha2b(type, Alpha[type],   NumTypes,          Alpha_num[type],   NNmodelRoot);
        readalpha3b(type, Alpha_a[type], NumTypesCouples,   Alpha_a_num[type], NNmodelRoot);
        if (NumTypes > 1) {
            reademb2b(type, Coeff_2b[type], NumTypes,        Alpha_num[type],   NNmodelRoot);
            reademb3b(type, Coeff_3b[type], NumTypesCouples, Alpha_a_num[type], NNmodelRoot);
        } else {
            fill_with_ones(Coeff_2b[0], Alpha_num[0]);
            fill_with_ones(Coeff_3b[0], Alpha_a_num[0]);
        }
    }
    for (int type = 0; type < NumTypes; type++) {
        cudaMemcpy(Alpha_d[type],    Alpha[type],    NumTypes * Alpha_num[type] * sizeof(double),        cudaMemcpyHostToDevice);
        cudaMemcpy(Alpha_a_d[type],  Alpha_a[type],  3 * NumTypesCouples * Alpha_a_num[type] * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Coeff_2b_d[type], Coeff_2b[type], NumTypes * Alpha_num[type] * sizeof(double),        cudaMemcpyHostToDevice);
        cudaMemcpy(Coeff_3b_d[type], Coeff_3b[type], NumTypesCouples * Alpha_a_num[type] * sizeof(double), cudaMemcpyHostToDevice);
    }
    printf("AlphaNes: Parameters saved on GPUs\n");
    fflush(stdout);

    int *dimAFSall = (int *)calloc(NumTypes, sizeof(int));
    dimAFSall_tot  = 0;
    for (int type = 0; type < NumTypes; type++)
        dimAFSall[type] = (Alpha_num[type] + Alpha_a_num[type]) * Tipos[type];
    for (int type = 0; type < NumTypes; type++)
        dimAFSall_tot += dimAFSall[type];
    AFs_all_d = createIrregularMatrix2D_CUDA(NumTypes, dimAFSall, 1);
    free(dimAFSall);
}

/* ===================================================================== */
/*  Constructor_MLP_Model                                                 */
/*  Replaces Constructor_TensorFlow_Model.                                */
/*  Allocates gradient buffers only; the StafMlp is set externally.      */
/* ===================================================================== */
void Constructor_MLP_Model()
{
    if (g_staf_mlp == NULL) {
        fprintf(stderr,
            "nn_nn_mlp: staf_jmd_set_mlp() must be called before "
            "initializenn_()\n");
        exit(1);
    }

    int *dimGrad = (int *)calloc(NumTypes, sizeof(int));
    for (int k = 0; k < NumTypes; k++)
        dimGrad[k] = (Alpha_num[k] + Alpha_a_num[k]) * Tipos[k];

    Gradients   = IrregularMatrix2Ddouble(NumTypes, dimGrad, 1.);
    Gradients_d = createIrregularMatrix2D_CUDA(NumTypes, dimGrad, 1.);
    free(dimGrad);

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("AlphaNes (MLP): free GPU mem %zu / total %zu bytes\n",
           freeMem, totalMem);
    /* Reserve a generous heap for device-side mallocs if needed */
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, (size_t)(freeMem * 0.6));
}

/* ===================================================================== */
/*  initializenn_                                                         */
/* ===================================================================== */
extern "C" void initializenn_(FILE *config_file, int number_of_particles)
{
    N = number_of_particles;
    Constructor_Descriptors(config_file);
    printf("Initializenn: Descriptors correctly constructed and loaded!\n");
    Constructor_AFS();
    printf("Initializenn: AFs correctly constructed and loaded!\n");
    Constructor_MLP_Model();
    printf("Initializenn: MLP model initialised (libstaf / StafMlp)!\n");
}

/* ===================================================================== */
/*  Compute_Descriptors  (unchanged)                                      */
/* ===================================================================== */
void Compute_Descriptors(double *box_d_arg, double *pos_d_arg,
                          int *howmany_d_arg, int *with_d_arg)
{
    set_tensor_to_zero_int(Numtriplet_d[0], N * sizeof(int));

    fill_radial_launcher(Cutoff, Radial_Buffer, Cutoff_Angular, Angular_Buffer, N,
                         pos_d_arg, box_d_arg,
                         howmany_d_arg, with_d_arg,
                         Des2b_d[0], Intmap2b_d[0], Der2b_d[0],
                         Des3bsupp_d[0], Der3bsupp_d[0], 1, Numtriplet_d[0],
                         Rs, coeffA, coeffB, coeffC,
                         Pow_alpha, Pow_beta);

    fill_angular_launcher(Cutoff, Radial_Buffer, Cutoff_Angular,
                          Angular_Buffer, N,
                          pos_d_arg, box_d_arg,
                          howmany_d_arg, with_d_arg,
                          Des3b_d[0], Intmap3b_d[0],
                          Des3bsupp_d[0], Der3b_d[0],
                          Der3bsupp_d[0], 1, Numtriplet_d[0]);
}

/* ===================================================================== */
/*  Compute_AFs_all  (unchanged)                                          */
/* ===================================================================== */
void Compute_AFs_all()
{
    set_tensor_to_zero_double(AFs_all_d[0], dimAFSall_tot);
    for (int type = 0; type < NumTypes; type++) {
        compute_2bodyAFs(type);
        compute_3bodyAFs(type);
    }
}

void compute_2bodyAFs(int type)
{
    radialAFs_Launcher(Des2b_d[type], Radial_Buffer,
                       Alpha_d[type], Alpha_num[type], Alpha_a_num[type],
                       AFs_all_d[type], 1, Tipos[type],
                       Intmap2b_d[type], Coeff_2b_d[type], Type_map_d);
}

void compute_3bodyAFs(int type)
{
    angularAFs_Launcher(Des3bsupp_d[type], Des3b_d[type],
                        Radial_Buffer, Angular_Buffer,
                        AFs_all_d[type], 1, Tipos[type],
                        Intmap3b_d[type], Alpha_a_d[type],
                        Alpha_a_num[type], Alpha_num[type],
                        Coeff_3b_d[type], Type_map_d, Numtriplet_d[type]);
}

/* ===================================================================== */
/*  Compute_NNEnergyandGradient_all                                       */
/*                                                                        */
/*  Strategy: pack all types' AFs into one contiguous host buffer,       */
/*  call staf_mlp_eval once (precision=1 → double), then scatter         */
/*  per-type gradients back to device.                                    */
/* ===================================================================== */

/* Per-type atom count and AF width for the eval call. */
static int *s_n_atoms   = NULL;  /* [NumTypes] */
static int *s_n_af      = NULL;  /* [NumTypes] */
static double *s_af_buf = NULL;  /* packed AFs  (host, double) */
static double *s_gr_buf = NULL;  /* packed grads (host, double) */
static float  *s_af_f32 = NULL;  /* float staging when mlp is float32 */
static float  *s_gr_f32 = NULL;
static int     s_buf_total = 0;  /* total elements allocated */

static void ensure_eval_buffers()
{
    /* Recompute total needed size. */
    int total = 0;
    for (int t = 0; t < NumTypes; t++)
        total += Tipos[t] * (Alpha_num[t] + Alpha_a_num[t]);

    if (total > s_buf_total) {
        free(s_af_buf);
        free(s_gr_buf);
        free(s_af_f32);
        free(s_gr_f32);
        s_af_buf   = (double *)malloc(total * sizeof(double));
        s_gr_buf   = (double *)malloc(total * sizeof(double));
        s_af_f32   = (float *)malloc(total * sizeof(float));
        s_gr_f32   = (float *)malloc(total * sizeof(float));
        s_buf_total = total;
    }
    if (!s_n_atoms) {
        s_n_atoms = (int *)malloc(NumTypes * sizeof(int));
        s_n_af    = (int *)malloc(NumTypes * sizeof(int));
    }
    for (int t = 0; t < NumTypes; t++) {
        s_n_atoms[t] = Tipos[t];
        s_n_af[t]    = Alpha_num[t] + Alpha_a_num[t];
    }
}

void Compute_NNEnergyandGradient_all(double *energy)
{
    *energy = 0.0;

    ensure_eval_buffers();

    /* Pack AFs: copy each type's AFs from device to contiguous host region. */
    int offset = 0;
    for (int t = 0; t < NumTypes; t++) {
        int n_elem = Tipos[t] * (Alpha_num[t] + Alpha_a_num[t]);
        cudaMemcpy(s_af_buf + offset, AFs_all_d[t],
                   n_elem * sizeof(double), cudaMemcpyDeviceToHost);
        offset += n_elem;
    }

    /* Optional dump for Python↔JMD AF parity checks (STAF_DUMP_AF=path). */
    {
        const char *dump_path = getenv("STAF_DUMP_AF");
        if (dump_path && dump_path[0]) {
            FILE *df = fopen(dump_path, "wb");
            if (df) {
                fwrite(&NumTypes, sizeof(int), 1, df);
                for (int t = 0; t < NumTypes; t++) {
                    int n_af = Alpha_num[t] + Alpha_a_num[t];
                    fwrite(&Tipos[t], sizeof(int), 1, df);
                    fwrite(&n_af, sizeof(int), 1, df);
                }
                fwrite(s_af_buf, sizeof(double), (size_t)s_buf_total, df);
                fclose(df);
                fprintf(stderr, "nn_nn_mlp: dumped AFs (%d doubles) to %s\n",
                        s_buf_total, dump_path);
            }
        }
        const char *ime_path = getenv("STAF_DUMP_IME");
        if (ime_path && ime_path[0] && howmany_d && with_d) {
            /* Host already has Ime copy in calculateforces — dump Des2b[0] contiguous. */
            int n_des = N * Radial_Buffer;
            double *des_h = (double *)malloc((size_t)n_des * sizeof(double));
            int *how_h = (int *)malloc((size_t)N * sizeof(int));
            cudaMemcpy(des_h, Des2b_d[0], (size_t)n_des * sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(how_h, howmany_d, (size_t)N * sizeof(int),
                       cudaMemcpyDeviceToHost);
            FILE *df = fopen(ime_path, "wb");
            if (df) {
                fwrite(&N, sizeof(int), 1, df);
                fwrite(&Radial_Buffer, sizeof(int), 1, df);
                fwrite(how_h, sizeof(int), (size_t)N, df);
                fwrite(des_h, sizeof(double), (size_t)n_des, df);
                fclose(df);
                fprintf(stderr, "nn_nn_mlp: dumped IME/Des2b to %s\n", ime_path);
            }
            free(des_h);
            free(how_h);
        }
    }

    /* Zero gradient output buffer. */
    memset(s_gr_buf, 0, s_buf_total * sizeof(double));

    const int prec = staf_mlp_precision(g_staf_mlp);
    double e_out = 0.0;
    float e_f = 0.f;
    StafMlpEval ev;
    memset(&ev, 0, sizeof(ev));
    ev.n_type  = NumTypes;
    ev.n_atoms = s_n_atoms;
    ev.n_af    = s_n_af;

    int rc;
    if (prec == 0) {
      /* ORT float32 graph: stage doubles → float */
      for (int i = 0; i < s_buf_total; ++i) s_af_f32[i] = (float)s_af_buf[i];
      memset(s_gr_f32, 0, s_buf_total * sizeof(float));
      ev.af_f32 = s_af_f32;
      ev.energy_f32 = &e_f;
      ev.dE_daf_f32 = s_gr_f32;
      rc = staf_mlp_eval(g_staf_mlp, &ev);
      if (rc == 0) {
        e_out = (double)e_f;
        for (int i = 0; i < s_buf_total; ++i) s_gr_buf[i] = (double)s_gr_f32[i];
      }
    } else {
      ev.af_f64 = s_af_buf;
      ev.energy_f64 = &e_out;
      ev.dE_daf_f64 = s_gr_buf;
      rc = staf_mlp_eval(g_staf_mlp, &ev);
    }
    if (rc != 0) {
        fprintf(stderr, "nn_nn_mlp: staf_mlp_eval returned error %d\n", rc);
        exit(1);
    }
    *energy += e_out;

    /* Scatter gradients back to per-type device buffers. */
    offset = 0;
    for (int t = 0; t < NumTypes; t++) {
        int n_elem = Tipos[t] * (Alpha_num[t] + Alpha_a_num[t]);
        /* Keep host copy in Gradients[t] for debugging if needed. */
        memcpy(Gradients[t], s_gr_buf + offset, n_elem * sizeof(double));
        cudaMemcpy(Gradients_d[t], Gradients[t],
                   n_elem * sizeof(double), cudaMemcpyHostToDevice);
        offset += n_elem;
    }
}

/* Single-type wrapper (kept for API completeness; accumulates into *energy). */
void Compute_NNEnergyandGradient(int type, double *energy)
{
    /*
     * For simplicity, calling the full _all variant on a single-type system.
     * If NumTypes > 1 and you need per-type energy, call _all once externally.
     * This stub avoids code duplication.
     */
    (void)type;
    Compute_NNEnergyandGradient_all(energy);
}

/* ===================================================================== */
/*  Force computation kernels  (unchanged from original)                  */
/* ===================================================================== */

void Compute_Force_2b(int type)
{
    int prod = Tipos[type] * Radial_Buffer;
    computeforce_doublets_Launcher(Gradients_d[type], Des2b_d[type],
                                   Der2b_d[type], Intmap2b_d[type],
                                   Radial_Buffer, N, 1,
                                   Alpha_num[type], Alpha_a_num[type],
                                   Alpha_d[type], Coeff_2b_d[type], NumTypes,
                                   Tipos_d, type, Force_d, Type_map_d, prod,
                                   Virial_Diagonal_d, pos_d, box_d);
}

void Compute_Force_3b(int type)
{
    int prod = Tipos[type] * Angular_Buffer;
    computeforce_tripl_Launcher(Gradients_d[type],
                                Des3bsupp_d[type], Des3b_d[type],
                                Der3bsupp_d[type], Der3b_d[type],
                                Intmap2b_d[type], Intmap3b_d[type],
                                Radial_Buffer, Angular_Buffer, N, 1,
                                Alpha_a_num[type], Alpha_num[type],
                                Coeff_3b_d[type], NumTypes, Tipos_d,
                                type, Force_d, Numtriplet_d[type],
                                Alpha_a_d[type], Type_map_d, prod,
                                Virial_Diagonal_d, pos_d, box_d);
}

void Compute_Forces_all(vector *force, double *virial, vector *virial_diag)
{
    set_tensor_to_zero_double(Force_d, 3 * N);
    set_tensor_to_zero_double(Virial_Diagonal_d, 3);
    for (int type = 0; type < NumTypes; type++) {
        Compute_Force_2b(type);
        Compute_Force_3b(type);
    }
    cudaMemcpy(force,       Force_d,           N * 3 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(virial_diag, Virial_Diagonal_d, 3 * sizeof(double),     cudaMemcpyDeviceToHost);
    *virial = virial_diag->x + virial_diag->y + virial_diag->z;
}

/* ===================================================================== */
/*  deletetensor — no-op (no TF tensors)                                  */
/* ===================================================================== */
void deletetensor()
{
    /* Nothing to do: StafMlp owns its own internal allocations. */
}

/* ===================================================================== */
/*  calculateforces  (C extern, unchanged logic)                          */
/* ===================================================================== */
extern "C" void calculateforces(vector *pos, double *box,
                                 interactionmap *Ime,
                                 double *energy, vector *force,
                                 double *virial, vector *virialxyz)
{
    cudaMemcpy(pos_d, pos, N * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(box_d, box, 6 * sizeof(double),     cudaMemcpyHostToDevice);

    MAX_PARTICLE_CELLS = Radial_Buffer;
    int c_nx, c_ny, c_nz;
    celleCompute(N, box, pos_d, Cutoff,
                 &Cells, &Cells_howmany, &c_nx, &c_ny, &c_nz,
                 MAX_PARTICLE_CELLS);
    imeCompute(N, box_d, pos_d, Cutoff,
               Cells, Cells_howmany, c_nx, c_ny, c_nz,
               with_d, howmany_d, with_dist2_d,
               MAX_PARTICLE_CELLS, Radial_Buffer);

    /* Copy IME to host */
    cudaMemcpy(Ime->howmany, howmany_d,    N * sizeof(int),                       cudaMemcpyDeviceToHost);
    cudaMemcpy(Ime->rij2[0], with_dist2_d, N * Radial_Buffer * sizeof(double),   cudaMemcpyDeviceToHost);
    cudaMemcpy(Ime->with[0], with_d,       N * Radial_Buffer * sizeof(int),       cudaMemcpyDeviceToHost);

    Compute_Descriptors(box_d, pos_d, howmany_d, with_d);
    Compute_AFs_all();
    Compute_NNEnergyandGradient_all(energy);
    Compute_Forces_all(force, virial, virialxyz);
    deletetensor();
}

/* ===================================================================== */
/*  calculate_energies  (C extern)                                        */
/* ===================================================================== */
extern "C" void calculate_energies(vector *pos, double *box,
                                    interactionmap *Ime,
                                    double *energy, vector *force,
                                    double *virial, vector *virialxyz)
{
    (void)force; (void)virial; (void)virialxyz;

    cudaMemcpy(pos_d, pos, N * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(box_d, box, 6 * sizeof(double),     cudaMemcpyHostToDevice);

    MAX_PARTICLE_CELLS = Radial_Buffer;
    int c_nx, c_ny, c_nz;
    celleCompute(N, box, pos_d, Cutoff,
                 &Cells, &Cells_howmany, &c_nx, &c_ny, &c_nz,
                 MAX_PARTICLE_CELLS);
    imeCompute(N, box_d, pos_d, Cutoff,
               Cells, Cells_howmany, c_nx, c_ny, c_nz,
               with_d, howmany_d, with_dist2_d,
               MAX_PARTICLE_CELLS, Radial_Buffer);

    cudaMemcpy(Ime->howmany, howmany_d,    N * sizeof(int),              cudaMemcpyDeviceToHost);
    cudaMemcpy(Ime->rij2[0], with_dist2_d, N * Radial_Buffer * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Ime->with[0], with_d,       N * Radial_Buffer * sizeof(int),    cudaMemcpyDeviceToHost);

    Compute_Descriptors(box_d, pos_d, howmany_d, with_d);
    Compute_AFs_all();
    Compute_NNEnergyandGradient_all(energy);
    deletetensor();
}

/* ===================================================================== */
/*  nnDestructor — no TF sessions; free gradient buffers                  */
/* ===================================================================== */
extern "C" void nnDestructor()
{
    /* Smoke: skip freeing IrregularMatrix / CUDA AF buffers (allocator
     * layout is not a simple free()). Leak on process exit is OK for B1. */
    s_af_buf = NULL;
    s_gr_buf = NULL;
    s_n_atoms = NULL;
    s_n_af = NULL;
    s_buf_total = 0;
}

/* ===================================================================== */
/*  Debug helpers                                                          */
/* ===================================================================== */
void Alpha_check(int *data, int num)
{
    printf("Alpha_check: ");
    for (int i = 0; i < num; i++) {
        printf(" %d ", data[i]);
        fflush(stdout);
    }
}

void Alpha_checkDO(double *data, int num)
{
    printf("Alpha_checkDO: ");
    for (int i = 0; i < num; i++) {
        printf(" %g ", data[i]);
        fflush(stdout);
    }
}
