/*
 * nn_nn_mlp.cu
 *
 * Patched from neuralmdGPU/DEV/CG_and_WCA_LJ2_inter/src/nn_nn.cu:
 *   - Removed TensorFlow C API
 *   - Constructor_TensorFlow_Model → Constructor_MLP_Model + staf_mlp_eval
 *   - KEEP dual cutoff Rc_inter/Rs_inter/Ra_inter and Map_intra /
 *     Color_type_map / map_color_interaction
 *   - STAF-only forces (no WCA / LJ)
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "staf_mlp.h"

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

#define SQR(x) ((x) * (x))
#define Sqrt(x) (sqrt(x))
#define Power(x, n) (pow(x, n))

static int N;
static int N_all;
static int N_model;
static int cap_n_all;
static int Radial_Buffer;
static int Angular_Buffer;
static double Cutoff;
static double Cutoff_Angular;
static double Rc_inter, Ra_inter, Rs_inter;

static double *box_d, *inobox_d, *pos_d;
static int *howmany_d, *with_d;
static double *with_dist2_d;
static int *Cells;
static int *Cells_howmany;
static int MAX_PARTICLE_CELLS;

static int **Intmap2b_d;
static int **Intmap3b_d;
static double **Des2b_d;
static double **Des3b_d;
static double **Des3bsupp_d;
static double **Der2b_d;
static double **Der3bsupp_d;
static double **Der3b_d;
static int **Numtriplet_d;

static int *Color_type_map;
static int *Color_type_map_d;
static int *Color_type_map_model;
static int *Map_intra;
static int *Map_intra_d;
static int *Map_intra_model;
static int *Map_color_interaction;
static int *Map_color_interaction_d;

static double **Alpha;
static double **Alpha_a;
static double **Alpha_d;
static double **Alpha_a_d;
static int dimAFSall_tot;
static int *Alpha_num;
static int *Alpha_a_num;
static int *Alpha_num_d;
static int *Alpha_a_num_d;

static double **Coeff_2b;
static double **Coeff_3b;
static double **Coeff_2b_d;
static double **Coeff_3b_d;

static double **AFs_all_d;
static int Num_NN;
static char NNmodelRoot[10000];
static char map_intra_file[10000];
static char color_type_map_file[10000];
static char map_color_interaction_file[10000];
static int Numcolors;

static double **Gradients, **Gradients_d;

static double Rs, coeffA_intra, coeffB_intra, coeffC_intra, Pow_alpha, Pow_beta;
static double coeffA_inter, coeffB_inter, coeffC_inter;
static double *Force_d;
static double *Virial_d;
static double *Virial_Diagonal_d;

static int g_ext_neigh = 0;
static int *howmany_h = NULL;
static int *with_h = NULL;
static int ext_maxneigh = 0;
static int cap_ext_centers = 0;
static int s_buf_total = 0;

static StafMlp *g_staf_mlp = NULL;

extern "C" void staf_jmd_set_mlp(void *mlp)
{
    g_staf_mlp = (StafMlp *)mlp;
}

extern "C" void staf_jmd_set_skip_pbc(int skip)
{
    (void)skip; /* CG kernels apply PBC internally; 1-rank path uses celle/ime. */
}

extern "C" void staf_jmd_set_external_neigh(const int *howmany_host,
                                            const int *with_host,
                                            int n_centers, int maxneigh)
{
    if (!howmany_host || !with_host || n_centers <= 0 || maxneigh <= 0)
        return;
    g_ext_neigh = 1;
    if (!howmany_h || !with_h || n_centers > cap_ext_centers) {
        free(howmany_h);
        free(with_h);
        howmany_h = (int *)malloc((size_t)n_centers * sizeof(int));
        with_h = (int *)malloc((size_t)n_centers * (size_t)Radial_Buffer *
                               sizeof(int));
        if (!howmany_h || !with_h) {
            fprintf(stderr, "staf_jmd_set_external_neigh: malloc failed\n");
            g_ext_neigh = 0;
            return;
        }
        cap_ext_centers = n_centers;
    }
    ext_maxneigh = maxneigh;
    (void)ext_maxneigh;
    for (int i = 0; i < n_centers; ++i) {
        int nn = howmany_host[i];
        if (nn < 0) nn = 0;
        if (nn > Radial_Buffer) nn = Radial_Buffer;
        if (nn > maxneigh) nn = maxneigh;
        howmany_h[i] = nn;
        for (int k = 0; k < Radial_Buffer; ++k) {
            with_h[i * Radial_Buffer + k] =
                (k < nn) ? with_host[i * maxneigh + k] : 0;
        }
    }
}

extern "C" void staf_jmd_clear_external_neigh(void)
{
    g_ext_neigh = 0;
}

extern "C" int staf_jmd_resize(int n_centers, int n_all,
                               const int *tipos_owned,
                               const int *type_map_all)
{
    (void)tipos_owned;
    if (n_centers < 0 || n_all < n_centers)
        return -1;
    /* Owned centers stay the model-sized origami (24 beads on 1-rank / one DD
     * subdomain that holds the dimer). Ghosts are extra slots for reverse_comm. */
    if (n_centers != N_model) {
        fprintf(stderr,
                "libstaf_cg: n_centers %d != model N %d "
                "(owned set must be the full origami for this runtime)\n",
                n_centers, N_model);
        return -1;
    }
    if (!Color_type_map_model || !Map_intra_model)
        return -1;

    N = n_centers;
    N_all = n_all;

    if (n_all > cap_n_all) {
        if (pos_d)
            cudaFree(pos_d);
        if (Force_d)
            cudaFree(Force_d);
        cudaMalloc((void **)&pos_d, 3 * (size_t)n_all * sizeof(double));
        cudaMalloc((void **)&Force_d, 3 * (size_t)n_all * sizeof(double));
        cap_n_all = n_all;
    }

    Color_type_map = (int *)realloc(Color_type_map, (size_t)n_all * sizeof(int));
    Map_intra = (int *)realloc(Map_intra, (size_t)n_all * sizeof(int));
    if (!Color_type_map || !Map_intra)
        return -1;
    for (int i = 0; i < n_all; ++i) {
        int bid = type_map_all ? type_map_all[i] : i;
        if (bid < 0 || bid >= N_model)
            return -1;
        Color_type_map[i] = Color_type_map_model[bid];
        Map_intra[i] = Map_intra_model[bid];
    }
    if (Color_type_map_d)
        cudaFree(Color_type_map_d);
    if (Map_intra_d)
        cudaFree(Map_intra_d);
    cudaMalloc((void **)&Color_type_map_d, (size_t)n_all * sizeof(int));
    cudaMalloc((void **)&Map_intra_d, (size_t)n_all * sizeof(int));
    cudaMemcpy(Color_type_map_d, Color_type_map, (size_t)n_all * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(Map_intra_d, Map_intra, (size_t)n_all * sizeof(int),
               cudaMemcpyHostToDevice);
    return 0;
}

extern "C" int staf_jmd_num_types(void)
{
    return Num_NN > 0 ? Num_NN : 1;
}

extern "C" int staf_jmd_radial_buffer(void)
{
    return Radial_Buffer;
}

int count_colors(int *color_type_map, int n)
{
    int numcolors = 0;
    for (int col = 0; col < 50; col++) {
        for (int k = 0; k < n; k++) {
            if (color_type_map[k] == col) {
                numcolors = numcolors + 1;
                break;
            }
        }
    }
    return numcolors;
}

static void save_cutoff_intra(double rc)
{
    FILE *newfile = fopen("cutoff_curve_intra.dat", "w");
    double dx = rc / 1000.;
    double x = 0;
    for (int k = 0; k < 1000; k++) {
        x = x + dx;
        if (x < Rs)
            fprintf(newfile, "%g %g\n", x,
                    coeffA_intra / pow(x / Rs, Pow_alpha) +
                        coeffB_intra / pow(x / Rs, Pow_beta) + coeffC_intra);
        else
            fprintf(newfile, "%g %g\n", x, 0.5 * (1 + cos(M_PI * x / rc)));
    }
    fclose(newfile);
}

static void save_cutoff_inter(double rc)
{
    FILE *newfile = fopen("cutoff_curve_inter.dat", "w");
    double dx = rc / 1000.;
    double x = 0;
    for (int k = 0; k < 1000; k++) {
        x = x + dx;
        if (x < Rs_inter)
            fprintf(newfile, "%g %g\n", x,
                    coeffA_inter / pow(x / Rs_inter, Pow_alpha) +
                        coeffB_inter / pow(x / Rs_inter, Pow_beta) +
                        coeffC_inter);
        else
            fprintf(newfile, "%g %g\n", x, 0.5 * (1 + cos(M_PI * x / rc)));
    }
    fclose(newfile);
}

static void construct_repulsion_intra(void)
{
    double alpha = 1.;
    double beta = -30.;
    Pow_alpha = alpha;
    Pow_beta = beta;
    double rs = Rs;
    double rc = Cutoff;
    double f = 0.5 * (cos(M_PI * rs / rc) + 1);
    double f1 = -0.5 * M_PI / rc * sin(M_PI * rs / rc);
    double f2 = -0.5 * SQR(M_PI / rc) * cos(M_PI * rs / rc);

    coeffB_intra = (f1 * rs + f2 * SQR(rs) / (alpha + 1)) * (alpha + 1) / beta /
                   (beta - alpha);
    coeffA_intra = (f2 * SQR(rs) - coeffB_intra * beta * (beta + 1)) /
                   (alpha * (alpha + 1));
    coeffC_intra = f - coeffA_intra - coeffB_intra;
    save_cutoff_intra(rc);
}

static void construct_repulsion_inter(void)
{
    double alpha = 1.;
    double beta = -30.;
    Pow_alpha = alpha;
    Pow_beta = beta;
    double rs = Rs_inter;
    double rc = Rc_inter;
    double f = 0.5 * (cos(M_PI * rs / rc) + 1);
    double f1 = -0.5 * M_PI / rc * sin(M_PI * rs / rc);
    double f2 = -0.5 * SQR(M_PI / rc) * cos(M_PI * rs / rc);

    coeffB_inter = (f1 * rs + f2 * SQR(rs) / (alpha + 1)) * (alpha + 1) / beta /
                   (beta - alpha);
    coeffA_inter = (f2 * SQR(rs) - coeffB_inter * beta * (beta + 1)) /
                   (alpha * (alpha + 1));
    coeffC_inter = f - coeffA_inter - coeffB_inter;
    save_cutoff_inter(rc);
}

void Constructor_Descriptors(FILE *config_file)
{
    SearchTable *s = searchNew();
    searchInt("Buffer_angular_descriptors", &Angular_Buffer, s);
    searchDouble("Cutoff", &Cutoff, s);
    searchDouble("Hardcore_Cutoff", &Rs, s);
    searchDouble("Cutoff_Angular", &Cutoff_Angular, s);
    searchDouble("Cutoff_Inter", &Rc_inter, s);
    searchDouble("Hardcore_Cutoff_Inter", &Rs_inter, s);
    searchDouble("Cutoff_Angular_Inter", &Ra_inter, s);
    searchString("nn_export_dir", NNmodelRoot, s);
    searchString("file_map_intra", map_intra_file, s);
    searchString("file_color_type_map", color_type_map_file, s);
    searchString("file_map_color_interaction", map_color_interaction_file, s);
    searchInt("number_of_NN", &Num_NN, s);
    searchInt("Buffer_radial_descriptors", &Radial_Buffer, s);
    searchFile(config_file, s);
    searchFree(s);

    printf("\n Inter cutoffs are radial %lf hard %lf angular %lf\n", Rc_inter,
           Rs_inter, Ra_inter);

    construct_repulsion_intra();
    construct_repulsion_inter();

    init_block_dim(Radial_Buffer);
    init_block_dim_ang(Angular_Buffer);

    Color_type_map = (int *)calloc((size_t)N, sizeof(int));
    cudaMalloc((void **)&Color_type_map_d, (size_t)N * sizeof(int));
    read_typemap(Color_type_map, N, color_type_map_file);
    cudaMemcpy(Color_type_map_d, Color_type_map, (size_t)N * sizeof(int),
               cudaMemcpyHostToDevice);

    Map_intra = (int *)calloc((size_t)N, sizeof(int));
    cudaMalloc((void **)&Map_intra_d, (size_t)N * sizeof(int));
    read_typemap(Map_intra, N, map_intra_file);
    cudaMemcpy(Map_intra_d, Map_intra, (size_t)N * sizeof(int),
               cudaMemcpyHostToDevice);

    N_model = N;
    N_all = N;
    cap_n_all = N;
    Color_type_map_model = (int *)malloc((size_t)N * sizeof(int));
    Map_intra_model = (int *)malloc((size_t)N * sizeof(int));
    if (!Color_type_map_model || !Map_intra_model) {
        fprintf(stderr, "libstaf_cg: map model copy malloc failed\n");
        exit(1);
    }
    memcpy(Color_type_map_model, Color_type_map, (size_t)N * sizeof(int));
    memcpy(Map_intra_model, Map_intra, (size_t)N * sizeof(int));

    Numcolors = count_colors(Color_type_map, N);
    printf("AlphaNes: Found %d colors of vertex ORIGAMI\n", Numcolors);
    fflush(stdout);
    Map_color_interaction = (int *)calloc((size_t)Numcolors, sizeof(int));
    cudaMalloc((void **)&Map_color_interaction_d,
               (size_t)Numcolors * sizeof(int));
    read_typemap(Map_color_interaction, Numcolors, map_color_interaction_file);
    cudaMemcpy(Map_color_interaction_d, Map_color_interaction,
               (size_t)Numcolors * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&box_d, 6 * sizeof(double));
    cudaMalloc((void **)&inobox_d, 6 * sizeof(double));
    cudaMalloc((void **)&pos_d, 3 * (size_t)N * sizeof(double));
    cudaMalloc((void **)&howmany_d, (size_t)N * sizeof(int));
    cudaMalloc((void **)&with_d, (size_t)N * (size_t)Radial_Buffer * sizeof(int));

    MAX_PARTICLE_CELLS = Radial_Buffer;
    cudaMalloc(&with_dist2_d,
               (size_t)N * (size_t)Radial_Buffer * sizeof(double));

    int *tipos = (int *)calloc(1, sizeof(int));
    tipos[0] = N;
    Intmap2b_d = createIrregularMatrix2D_CUDA_int(1, tipos, (Radial_Buffer + 1));
    Intmap3b_d = createIrregularMatrix2D_CUDA_int(1, tipos, (Angular_Buffer * 2));
    Des2b_d = createIrregularMatrix2D_CUDA(1, tipos, Radial_Buffer);
    Des3b_d = createIrregularMatrix2D_CUDA(1, tipos, Angular_Buffer);
    Des3bsupp_d = createIrregularMatrix2D_CUDA(1, tipos, Radial_Buffer);
    Der2b_d = createIrregularMatrix2D_CUDA(1, tipos, (Radial_Buffer * 3));
    Der3bsupp_d = createIrregularMatrix2D_CUDA(1, tipos, (Radial_Buffer * 3));
    Der3b_d = createIrregularMatrix2D_CUDA(1, tipos, (Angular_Buffer * 2 * 3));
    Numtriplet_d = createIrregularMatrix2D_CUDA_int(1, tipos, 1);
    free(tipos);

    cudaMalloc((void **)&Force_d, 3 * (size_t)N * sizeof(double));
    cudaMalloc((void **)&Virial_d, sizeof(double));
    cudaMalloc((void **)&Virial_Diagonal_d, 3 * sizeof(double));
}

void Constructor_AFS(void)
{
    Alpha_num = (int *)calloc((size_t)Num_NN, sizeof(int));
    Alpha_a_num = (int *)calloc((size_t)Num_NN, sizeof(int));
    cudaMalloc((void **)&Alpha_num_d, (size_t)Num_NN * sizeof(int));
    cudaMalloc((void **)&Alpha_a_num_d, (size_t)Num_NN * sizeof(int));

    readAlpha_num(NNmodelRoot, Alpha_num, Alpha_a_num, Num_NN);
    printf("AlphaNes: Number of radial AFS\n");
    print_data_int(Alpha_num, Num_NN);
    printf("AlphaNes: Number of angular AFS\n");
    print_data_int(Alpha_a_num, Num_NN);

    cudaMemcpy(Alpha_num_d, Alpha_num, (size_t)Num_NN * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(Alpha_a_num_d, Alpha_a_num, (size_t)Num_NN * sizeof(int),
               cudaMemcpyHostToDevice);

    Alpha = IrregularMatrix2Ddouble(Num_NN, Alpha_num, 3);
    Alpha_a = IrregularMatrix2Ddouble(Num_NN, Alpha_a_num, 3 * 6);
    Coeff_2b = IrregularMatrix2Ddouble(Num_NN, Alpha_num, 3);
    Coeff_3b = IrregularMatrix2Ddouble(Num_NN, Alpha_a_num, 6);

    Alpha_d = createIrregularMatrix2D_CUDA(Num_NN, Alpha_num, 3);
    Alpha_a_d = createIrregularMatrix2D_CUDA(Num_NN, Alpha_a_num, 3 * 6);
    Coeff_2b_d = createIrregularMatrix2D_CUDA(Num_NN, Alpha_num, 3);
    Coeff_3b_d = createIrregularMatrix2D_CUDA(Num_NN, Alpha_a_num, 6);

    for (int nnindex = 0; nnindex < Num_NN; nnindex++) {
        readalpha2b(nnindex, Alpha[nnindex], 3, Alpha_num[nnindex], NNmodelRoot);
        readalpha3b(nnindex, Alpha_a[nnindex], 6, Alpha_a_num[nnindex],
                    NNmodelRoot);
        reademb2b(nnindex, Coeff_2b[nnindex], 3, Alpha_num[nnindex], NNmodelRoot);
        reademb3b(nnindex, Coeff_3b[nnindex], 6, Alpha_a_num[nnindex],
                  NNmodelRoot);
    }
    for (int nnindex = 0; nnindex < Num_NN; nnindex++) {
        cudaMemcpy(Alpha_d[nnindex], Alpha[nnindex],
                   3 * (size_t)Alpha_num[nnindex] * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(Alpha_a_d[nnindex], Alpha_a[nnindex],
                   3 * 6 * (size_t)Alpha_a_num[nnindex] * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(Coeff_2b_d[nnindex], Coeff_2b[nnindex],
                   3 * (size_t)Alpha_num[nnindex] * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(Coeff_3b_d[nnindex], Coeff_3b[nnindex],
                   6 * (size_t)Alpha_a_num[nnindex] * sizeof(double),
                   cudaMemcpyHostToDevice);
    }
    printf("AlphaNes: AFS parameters saved on GPUs\n");
    fflush(stdout);

    int *dimAFSall = (int *)calloc((size_t)Num_NN, sizeof(int));
    dimAFSall_tot = 0;
    for (int nnindex = 0; nnindex < Num_NN; nnindex++) {
        dimAFSall[nnindex] = (Alpha_num[nnindex] + Alpha_a_num[nnindex]) * N;
        dimAFSall_tot += dimAFSall[nnindex];
    }
    AFs_all_d = createIrregularMatrix2D_CUDA(Num_NN, dimAFSall, 1);
    free(dimAFSall);
}

void Constructor_MLP_Model(void)
{
    if (g_staf_mlp == NULL) {
        fprintf(stderr,
                "nn_nn_mlp: staf_jmd_set_mlp() must be called before "
                "initializenn_()\n");
        exit(1);
    }

    int *dimGrad = (int *)calloc((size_t)Num_NN, sizeof(int));
    for (int k = 0; k < Num_NN; k++)
        dimGrad[k] = (Alpha_num[k] + Alpha_a_num[k]) * N;

    Gradients = IrregularMatrix2Ddouble(Num_NN, dimGrad, 1.);
    Gradients_d = createIrregularMatrix2D_CUDA(Num_NN, dimGrad, 1.);
    free(dimGrad);

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("AlphaNes (MLP CG): free GPU mem %zu / total %zu bytes\n", freeMem,
           totalMem);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, (size_t)(freeMem * 0.6));
}

extern "C" void initializenn_(FILE *config_file, int number_of_particles)
{
    N = number_of_particles;
    Constructor_Descriptors(config_file);
    printf("Initializenn: Descriptors correctly constructed and loaded!\n");
    Constructor_AFS();
    printf("Initializenn: AFs correctly constructed and loaded!\n");
    Constructor_MLP_Model();
    printf("Initializenn: MLP model initialised (libstaf_cg / StafMlp)!\n");
}

void Compute_Descriptors(double *box_d_arg, double *pos_d_arg,
                         int *howmany_d_arg, int *with_d_arg)
{
    set_tensor_to_zero_int(Numtriplet_d[0], N);
    fill_radial_launcher(Cutoff, Radial_Buffer, Cutoff_Angular, Angular_Buffer,
                         N, pos_d_arg, box_d_arg, howmany_d_arg, with_d_arg,
                         Des2b_d[0], Intmap2b_d[0], Der2b_d[0], Des3bsupp_d[0],
                         Der3bsupp_d[0], 1, Numtriplet_d[0], Rs, coeffA_intra,
                         coeffB_intra, coeffC_intra, coeffA_inter, coeffB_inter,
                         coeffC_inter, Pow_alpha, Pow_beta, Rc_inter, Rs_inter,
                         Ra_inter, Map_intra_d, Color_type_map_d);
    fill_angular_launcher(Cutoff, Radial_Buffer, Cutoff_Angular, Angular_Buffer,
                          N, pos_d_arg, box_d_arg, howmany_d_arg, with_d_arg,
                          Des3b_d[0], Intmap3b_d[0], Des3bsupp_d[0], Der3b_d[0],
                          Der3bsupp_d[0], 1, Numtriplet_d[0], Rc_inter,
                          Rs_inter, Ra_inter, Map_intra_d, Color_type_map_d);
}

void Compute_AFs_all(void)
{
    set_tensor_to_zero_double(AFs_all_d[0], dimAFSall_tot);
    for (int nnindex = 0; nnindex < Num_NN; nnindex++) {
        compute_2bodyAFs(nnindex);
        compute_3bodyAFs(nnindex);
    }
}

void compute_2bodyAFs(int nnindex)
{
    radialAFs_Launcher(Des2b_d[0], Radial_Buffer, Alpha_d[nnindex],
                       Alpha_num[nnindex], Alpha_a_num[nnindex],
                       AFs_all_d[nnindex], 1, N, Intmap2b_d[0],
                       Coeff_2b_d[nnindex], Color_type_map_d, Map_intra_d,
                       Map_color_interaction_d);
}

void compute_3bodyAFs(int nnindex)
{
    angularAFs_Launcher(Des3bsupp_d[0], Des3b_d[0], Radial_Buffer,
                        Angular_Buffer, AFs_all_d[nnindex], 1, N, Intmap3b_d[0],
                        Alpha_a_d[nnindex], Alpha_a_num[nnindex],
                        Alpha_num[nnindex], Coeff_3b_d[nnindex],
                        Color_type_map_d, Numtriplet_d[0], Map_intra_d,
                        Map_color_interaction_d);
}

static int *s_n_atoms = NULL;
static int *s_n_af = NULL;
static double *s_af_buf = NULL;
static double *s_gr_buf = NULL;
static float *s_af_f32 = NULL;
static float *s_gr_f32 = NULL;

static void ensure_eval_buffers(void)
{
    int total = 0;
    for (int t = 0; t < Num_NN; t++)
        total += N * (Alpha_num[t] + Alpha_a_num[t]);

    if (total > s_buf_total) {
        free(s_af_buf);
        free(s_gr_buf);
        free(s_af_f32);
        free(s_gr_f32);
        s_af_buf = (double *)malloc((size_t)total * sizeof(double));
        s_gr_buf = (double *)malloc((size_t)total * sizeof(double));
        s_af_f32 = (float *)malloc((size_t)total * sizeof(float));
        s_gr_f32 = (float *)malloc((size_t)total * sizeof(float));
        s_buf_total = total;
    }
    if (!s_n_atoms) {
        s_n_atoms = (int *)malloc((size_t)Num_NN * sizeof(int));
        s_n_af = (int *)malloc((size_t)Num_NN * sizeof(int));
    }
    for (int t = 0; t < Num_NN; t++) {
        s_n_atoms[t] = N;
        s_n_af[t] = Alpha_num[t] + Alpha_a_num[t];
    }
}

void Compute_NNEnergyandGradient_all(double *energy)
{
    *energy = 0.0;
    ensure_eval_buffers();

    int offset = 0;
    for (int t = 0; t < Num_NN; t++) {
        int n_elem = N * (Alpha_num[t] + Alpha_a_num[t]);
        cudaMemcpy(s_af_buf + offset, AFs_all_d[t],
                   (size_t)n_elem * sizeof(double), cudaMemcpyDeviceToHost);
        offset += n_elem;
    }

    {
        const char *dump_path = getenv("STAF_DUMP_AF");
        if (dump_path && dump_path[0]) {
            FILE *df = fopen(dump_path, "wb");
            if (df) {
                fwrite(&Num_NN, sizeof(int), 1, df);
                for (int t = 0; t < Num_NN; t++) {
                    int n_af = Alpha_num[t] + Alpha_a_num[t];
                    fwrite(&N, sizeof(int), 1, df);
                    fwrite(&n_af, sizeof(int), 1, df);
                }
                fwrite(s_af_buf, sizeof(double), (size_t)s_buf_total, df);
                fclose(df);
                fprintf(stderr, "nn_nn_mlp CG: dumped AFs (%d doubles) to %s\n",
                        s_buf_total, dump_path);
            }
        }
    }

    memset(s_gr_buf, 0, (size_t)s_buf_total * sizeof(double));

    const int prec = staf_mlp_precision(g_staf_mlp);
    double e_out = 0.0;
    float e_f = 0.f;
    StafMlpEval ev;
    memset(&ev, 0, sizeof(ev));
    ev.n_type = Num_NN;
    ev.n_atoms = s_n_atoms;
    ev.n_af = s_n_af;

    int rc;
    if (prec == 0) {
        for (int i = 0; i < s_buf_total; ++i)
            s_af_f32[i] = (float)s_af_buf[i];
        memset(s_gr_f32, 0, (size_t)s_buf_total * sizeof(float));
        ev.af_f32 = s_af_f32;
        ev.energy_f32 = &e_f;
        ev.dE_daf_f32 = s_gr_f32;
        rc = staf_mlp_eval(g_staf_mlp, &ev);
        if (rc == 0) {
            e_out = (double)e_f;
            for (int i = 0; i < s_buf_total; ++i)
                s_gr_buf[i] = (double)s_gr_f32[i];
        }
    } else {
        ev.af_f64 = s_af_buf;
        ev.energy_f64 = &e_out;
        ev.dE_daf_f64 = s_gr_buf;
        rc = staf_mlp_eval(g_staf_mlp, &ev);
    }
    if (rc != 0) {
        fprintf(stderr, "nn_nn_mlp CG: staf_mlp_eval returned error %d\n", rc);
        exit(1);
    }
    *energy += e_out;

    offset = 0;
    for (int t = 0; t < Num_NN; t++) {
        int n_elem = N * (Alpha_num[t] + Alpha_a_num[t]);
        memcpy(Gradients[t], s_gr_buf + offset, (size_t)n_elem * sizeof(double));
        cudaMemcpy(Gradients_d[t], Gradients[t],
                   (size_t)n_elem * sizeof(double), cudaMemcpyHostToDevice);
        offset += n_elem;
    }
}

void Compute_NNEnergyandGradient(int nnindex, double *energy)
{
    (void)nnindex;
    Compute_NNEnergyandGradient_all(energy);
}

void Compute_Force_2b(int nnindex)
{
    int prod = N * Radial_Buffer;
    computeforce_doublets_Launcher(
        Gradients_d[nnindex], Des2b_d[0], Der2b_d[0], Intmap2b_d[0],
        Radial_Buffer, N, 1, Alpha_num[nnindex], Alpha_a_num[nnindex],
        Alpha_d[nnindex], Coeff_2b_d[nnindex], Force_d, Color_type_map_d, prod,
        Virial_Diagonal_d, pos_d, box_d, Map_intra_d, Map_color_interaction_d,
        N_all);
}

void Compute_Force_3b(int nnindex)
{
    int prod = N * Angular_Buffer;
    computeforce_tripl_Launcher(
        Gradients_d[nnindex], Des3bsupp_d[0], Des3b_d[0], Der3bsupp_d[0],
        Der3b_d[0], Intmap2b_d[0], Intmap3b_d[0], Radial_Buffer, Angular_Buffer,
        N, 1, Alpha_a_num[nnindex], Alpha_num[nnindex], Coeff_3b_d[nnindex],
        Force_d, Numtriplet_d[0], Alpha_a_d[nnindex], Color_type_map_d, prod,
        Virial_Diagonal_d, pos_d, box_d, Map_intra_d, Map_color_interaction_d,
        N_all);
}

void Compute_Forces_all(vector *force, double *virial, vector *virial_diag)
{
    set_tensor_to_zero_double(Force_d, 3 * N_all);
    set_tensor_to_zero_double(Virial_Diagonal_d, 3);
    for (int nnindex = 0; nnindex < Num_NN; nnindex++) {
        Compute_Force_2b(nnindex);
        Compute_Force_3b(nnindex);
    }
    cudaMemcpy(force, Force_d, (size_t)N_all * 3 * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(virial_diag, Virial_Diagonal_d, 3 * sizeof(double),
               cudaMemcpyDeviceToHost);
    *virial = virial_diag->x + virial_diag->y + virial_diag->z;
}

void deletetensor(void) {}

extern "C" void calculateforces(vector *pos, double *box, interactionmap *Ime,
                                 double *energy, vector *force, double *virial,
                                 vector *virialxyz)
{
    cudaMemcpy(pos_d, pos, (size_t)N_all * 3 * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(box_d, box, 6 * sizeof(double), cudaMemcpyHostToDevice);

    if (g_ext_neigh) {
        cudaMemcpy(howmany_d, howmany_h, (size_t)N * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(with_d, with_h,
                   (size_t)N * (size_t)Radial_Buffer * sizeof(int),
                   cudaMemcpyHostToDevice);
    } else {
        MAX_PARTICLE_CELLS = Radial_Buffer;
        int c_nx, c_ny, c_nz;
        celleCompute(N, box, pos_d, Cutoff, &Cells, &Cells_howmany, &c_nx,
                     &c_ny, &c_nz, MAX_PARTICLE_CELLS);
        imeCompute(N, box_d, pos_d, Cutoff, Cells, Cells_howmany, c_nx, c_ny,
                   c_nz, with_d, howmany_d, with_dist2_d, MAX_PARTICLE_CELLS,
                   Radial_Buffer);
        if (Ime) {
            cudaMemcpy(Ime->howmany, howmany_d, (size_t)N * sizeof(int),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(Ime->rij2[0], with_dist2_d,
                       (size_t)N * Radial_Buffer * sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(Ime->with[0], with_d,
                       (size_t)N * Radial_Buffer * sizeof(int),
                       cudaMemcpyDeviceToHost);
        }
    }

    Compute_Descriptors(box_d, pos_d, howmany_d, with_d);
    Compute_AFs_all();
    Compute_NNEnergyandGradient_all(energy);
    Compute_Forces_all(force, virial, virialxyz);
    deletetensor();
}

extern "C" void calculate_energies(vector *pos, double *box,
                                    interactionmap *Ime, double *energy,
                                    vector *force, double *virial,
                                    vector *virialxyz)
{
    (void)force;
    (void)virial;
    (void)virialxyz;
    cudaMemcpy(pos_d, pos, (size_t)N_all * 3 * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(box_d, box, 6 * sizeof(double), cudaMemcpyHostToDevice);

    if (g_ext_neigh) {
        cudaMemcpy(howmany_d, howmany_h, (size_t)N * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(with_d, with_h,
                   (size_t)N * (size_t)Radial_Buffer * sizeof(int),
                   cudaMemcpyHostToDevice);
    } else {
        MAX_PARTICLE_CELLS = Radial_Buffer;
        int c_nx, c_ny, c_nz;
        celleCompute(N, box, pos_d, Cutoff, &Cells, &Cells_howmany, &c_nx,
                     &c_ny, &c_nz, MAX_PARTICLE_CELLS);
        imeCompute(N, box_d, pos_d, Cutoff, Cells, Cells_howmany, c_nx, c_ny,
                   c_nz, with_d, howmany_d, with_dist2_d, MAX_PARTICLE_CELLS,
                   Radial_Buffer);
        if (Ime) {
            cudaMemcpy(Ime->howmany, howmany_d, (size_t)N * sizeof(int),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(Ime->rij2[0], with_dist2_d,
                       (size_t)N * Radial_Buffer * sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(Ime->with[0], with_d,
                       (size_t)N * Radial_Buffer * sizeof(int),
                       cudaMemcpyDeviceToHost);
        }
    }

    Compute_Descriptors(box_d, pos_d, howmany_d, with_d);
    Compute_AFs_all();
    Compute_NNEnergyandGradient_all(energy);
    deletetensor();
}

extern "C" void nnDestructor(void)
{
    s_af_buf = NULL;
    s_gr_buf = NULL;
    s_n_atoms = NULL;
    s_n_af = NULL;
    s_buf_total = 0;
}

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
