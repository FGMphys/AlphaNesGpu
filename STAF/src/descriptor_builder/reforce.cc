#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <ctype.h>
#include <mutex>
#include <unordered_map>
#include <memory>
#include "staf_real.h"

#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"

#include "celle_gpu.h"
#include <cuda_runtime.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#define PI real(3.141592654)
#define SQR(x) ((x)*(x))
#define Power(x,n) (staf_pow((real)(x),(real)(n)))
/* Must match BLOCK_DIM in celle_gpu.cu.cc (threads per cell). */
#define BLOCK_DIM_SAFE 256

/* Scalars shared by all devices (same model geometry). Device buffers are per-GPU. */
struct StafDescriptorConfig {
  int Radbuff = 0;
  int Angbuff = 0;
  int N_max = 0;
  int max_batch = 0;
  real R_c = 0, Rs = 0, R_a = 0;
  real coeffA = 0, coeffB = 0, coeffC = 0, Pow_alpha = 0, Pow_beta = 0;
  bool ready = false;
};

struct StafDescriptorCtx {
  StafDescriptorConfig cfg;
  real* Full_box = nullptr;
  real* nowinobox = nullptr;
  real* nowinobox_d = nullptr;
  real* with_dist2_d = nullptr;
  int* Cells = nullptr;
  int* Cells_howmany = nullptr;
  int cells_capacity_num = 0;
  int cells_capacity_mpc = 0;
  int MAX_PARTICLE_CELLS = 0;
  int* howmany_d = nullptr;
  int* with_d = nullptr;
  int* code_ret_d = nullptr;
  int* code_ret = nullptr;
  real* nowinopos_d = nullptr;
  int device_id = -1;

  void allocate(int device) {
    device_id = device;
    cudaSetDevice(device);
    MAX_PARTICLE_CELLS = cfg.N_max / 3;
    if (MAX_PARTICLE_CELLS < 1) MAX_PARTICLE_CELLS = 1;
    if (MAX_PARTICLE_CELLS > BLOCK_DIM_SAFE) MAX_PARTICLE_CELLS = BLOCK_DIM_SAFE;
    int nf = cfg.max_batch;
    int N = cfg.N_max;
    Full_box = (real*)calloc(nf * 6, sizeof(real));
    nowinobox = (real*)calloc(nf * 6, sizeof(real));
    cudaMalloc(&nowinobox_d, nf * 6 * sizeof(real));
    cudaMalloc(&with_dist2_d, nf * N * cfg.Radbuff * sizeof(real));
    cudaMalloc(&howmany_d, nf * N * sizeof(int));
    cudaMalloc(&with_d, nf * N * cfg.Radbuff * sizeof(int));
    cudaMalloc(&nowinopos_d, nf * N * 3 * sizeof(real));
    cudaMalloc(&code_ret_d, sizeof(int));
    code_ret = (int*)calloc(1, sizeof(int));
    Cells = nullptr;
    Cells_howmany = nullptr;
    cells_capacity_num = 0;
    cells_capacity_mpc = 0;
  }

  ~StafDescriptorCtx() {
    if (device_id >= 0) cudaSetDevice(device_id);
    free(Full_box);
    free(nowinobox);
    free(code_ret);
    if (nowinobox_d) cudaFree(nowinobox_d);
    if (with_dist2_d) cudaFree(with_dist2_d);
    if (howmany_d) cudaFree(howmany_d);
    if (with_d) cudaFree(with_d);
    if (nowinopos_d) cudaFree(nowinopos_d);
    if (code_ret_d) cudaFree(code_ret_d);
    if (Cells) cudaFree(Cells);
    if (Cells_howmany) cudaFree(Cells_howmany);
  }
};

static std::mutex g_desc_mu;
static StafDescriptorConfig g_desc_cfg;
static std::unordered_map<int, std::unique_ptr<StafDescriptorCtx>> g_desc_ctx;

static void fill_repulsion(StafDescriptorConfig* c) {
  real alpha = 1.;
  real beta = -30.;
  c->Pow_alpha = alpha;
  c->Pow_beta = beta;
  real rs = c->Rs;
  real rc = c->R_c;
  real f = 0.5 * (staf_cos(PI * rs / rc) + 1);
  real f1 = -0.5 * PI / rc * staf_sin(PI * rs / rc);
  real f2_red = -0.5 * SQR(PI / rc) * staf_cos(PI * rs / rc) * SQR(rs);
  real gamma_red = 1. / (alpha - beta) * alpha - 1;
  real delta_red =
      1. / (alpha - beta) * (f * (alpha - beta) - f1 * rs - f * alpha);
  real eta_red = -alpha / (alpha - beta);
  real epsilon_red = 1. / (alpha - beta) * (rs * f1 + alpha * f);
  real c2_red = alpha * (alpha + 1) * delta_red + beta * (beta + 1) * epsilon_red;
  real c1_red = alpha * (alpha + 1) * gamma_red + beta * (beta + 1) * eta_red;
  c->coeffC = (f2_red - c2_red) / c1_red;
  real eta = -alpha * Power(rs, beta) / (alpha - beta);
  real epsilon = Power(rs, beta) / (alpha - beta) * (rs * f1 + alpha * f);
  c->coeffB = eta * c->coeffC + epsilon;
  real gamma = Power(rs, alpha) / (alpha - beta) * alpha - Power(rs, alpha);
  real delta =
      Power(rs, alpha) / (alpha - beta) * (f * (alpha - beta) - f1 * rs - f * alpha);
  c->coeffA = gamma * c->coeffC + delta;

  FILE* newfile = fopen("cutoff_curve.dat", "w");
  real dx = rc / 1000.;
  real x = 0;
  for (int k = 0; k < 1000; k++) {
    x = x + dx;
    if (x < rs) {
      fprintf(newfile, "%g %g\n", x,
              c->coeffA / Power(x, c->Pow_alpha) +
                  c->coeffB / Power(x, c->Pow_beta) + c->coeffC);
    } else {
      fprintf(newfile, "%g %g\n", x, 0.5 * (1 + staf_cos(PI * x / rc)));
    }
  }
  fclose(newfile);
}

static StafDescriptorCtx* get_or_create_ctx(int device) {
  std::lock_guard<std::mutex> lock(g_desc_mu);
  if (!g_desc_cfg.ready) {
    fprintf(stderr, "STAF: ComputeDescriptorsLight before ConstructDescriptorsLight\n");
    exit(1);
  }
  auto it = g_desc_ctx.find(device);
  if (it != g_desc_ctx.end()) return it->second.get();
  auto ctx = std::unique_ptr<StafDescriptorCtx>(new StafDescriptorCtx());
  ctx->cfg = g_desc_cfg;
  ctx->allocate(device);
  StafDescriptorCtx* raw = ctx.get();
  g_desc_ctx.emplace(device, std::move(ctx));
  printf("STAF: Descriptor context created for CUDA device %d\n", device);
  return raw;
}

 void fill_radial_launcher(real R_c,int radbuff,real R_a,int angbuff,int N,
                       real* inopos_d,const real* box_d,
                       int *howmany_d,int *with_d,
                       real* descriptor_d,int* intmap2b_d,real* der2b_d,
                       real* des3bsupp_d,
                       real* der3bsupp_d, int nf,int* numtriplet_d,
                       real rs, real coeffa,real coeffb,real coeffc,real pow_alpha, real pow_beta, cudaStream_t stream);
 void fill_angular_launcher(real R_c,int radbuff,real R_a,int angbuff,int N,
                       real* inopos_d,const real* box_d,
                       int *howmany_d,int *with_d,
                       real* ang_descr_d,int* intmap3b_d,
                       real* des3bsupp_d,real* der3b_d,
                       real* der3bsupp_d, int nf,int* numtriplet_d, cudaStream_t stream);

void set_tensor_to_zero_int(int* tensor,int dimten, cudaStream_t stream);

void set_tensor_to_zero_real(real* tensor,int dimten, cudaStream_t stream);

void check_max_launcher(int* tensor,int dim,int maxval,int* resval, cudaStream_t stream);

using namespace tensorflow;

REGISTER_OP("ConstructDescriptorsLight")
    .Input("radial_cutoff: " STAF_TF_DTYPE)
    .Input("radial_buffer: int32")
    .Input("angular_buffer: int32")
    .Input("numpar: int32")
    .Input("boxer: " STAF_TF_DTYPE)
    .Input("rs: " STAF_TF_DTYPE)
    .Input("ra: " STAF_TF_DTYPE)
    .Input("max_batch: int32")
    .Output("exitcode: int32");

 class ConstructDescriptorsLightOp : public OpKernel {
     public:
      explicit ConstructDescriptorsLightOp(OpKernelConstruction* context) : OpKernel(context) {


      }

      void Compute(OpKernelContext* context) override {
           const Tensor& rcrad_T = context->input(0);
           const Tensor& radbuff_T = context->input(1);
           const Tensor& angbuff_T = context->input(2);
           const Tensor& numpar_T = context->input(3);
           const Tensor& box_T = context->input(4);
           const Tensor& rs_T = context->input(5);
           const Tensor& ra_T = context->input(6);
           const Tensor& max_batch_T = context->input(7);

           StafDescriptorConfig cfg;
           cfg.Rs = rs_T.flat<real>()(0);
           cfg.R_c = rcrad_T.flat<real>()(0);
           cfg.Radbuff = radbuff_T.flat<int>()(0);
           cfg.Angbuff = angbuff_T.flat<int>()(0);
           cfg.R_a = ra_T.flat<real>()(0);
           cfg.N_max = numpar_T.flat<int>()(0);
           cfg.max_batch = max_batch_T.flat<int>()(0);
           (void)box_T;

           printf("\nSTAF: Descriptor constructor found Rc %f\n", cfg.R_c);
           printf("          Ra %f Rs %f Radbuff %d Angbuff %d max_batch %d N_max %d\n",
                  cfg.R_a, cfg.Rs, cfg.Radbuff, cfg.Angbuff, cfg.max_batch, cfg.N_max);
           fill_repulsion(&cfg);
           cfg.ready = true;

           {
             std::lock_guard<std::mutex> lock(g_desc_mu);
             g_desc_cfg = cfg;
             /* Drop old device contexts so next Compute reallocates with new geometry. */
             g_desc_ctx.clear();
           }
           /* Eager alloc on current CUDA device (usually 0) for single-GPU path. */
           int dev = 0;
           cudaGetDevice(&dev);
           get_or_create_ctx(dev);
         }
    };
REGISTER_KERNEL_BUILDER(Name("ConstructDescriptorsLight").Device(DEVICE_CPU), ConstructDescriptorsLightOp);


REGISTER_OP("ComputeDescriptorsLight")
    .Input("positions: " STAF_TF_DTYPE)
    .Input("boxer: " STAF_TF_DTYPE)
    .Output("raddescr: " STAF_TF_DTYPE)
    .Output("angdescr: " STAF_TF_DTYPE)
    .Output("des3bsupp: " STAF_TF_DTYPE)
    .Output("intmap2b: int32")
    .Output("intmap3b: int32")
    .Output("der2b: " STAF_TF_DTYPE)
    .Output("der3b: " STAF_TF_DTYPE)
    .Output("der3bsupp: " STAF_TF_DTYPE)
    .Output("numtriplet: int32");


class ComputeDescriptorsLightOp : public OpKernel {
 public:
  explicit ComputeDescriptorsLightOp(OpKernelConstruction* context) : OpKernel(context) {


  }

  void Compute(OpKernelContext* context) override {
    const cudaStream_t stream = context->eigen_device<Eigen::GpuDevice>().stream();
    int device = 0;
    cudaGetDevice(&device);
    StafDescriptorCtx* ctx = get_or_create_ctx(device);
    const StafDescriptorConfig& C = ctx->cfg;

    const Tensor& positions_T = context->input(0);
    const Tensor& box_T = context->input(1);

    auto positions = positions_T.flat<real>();
    const real* nowpos_d = positions.data();
    const real* nowbox_d = box_T.flat<real>().data();

    int nf = box_T.shape().dim_size(0);
    int N = int(positions_T.shape().dim_size(1) / 3);

    cudaMemcpyAsync(ctx->Full_box, nowbox_d, sizeof(real) * nf * 6,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int fr = 0; fr < nf; fr++) {
      ctx->nowinobox[0 + fr * 6] = real(1.) / ctx->Full_box[0 + fr * 6];
      ctx->nowinobox[1 + fr * 6] =
          -ctx->Full_box[1 + fr * 6] /
          (ctx->Full_box[0 + fr * 6] * ctx->Full_box[3 + fr * 6]);
      ctx->nowinobox[2 + fr * 6] =
          (ctx->Full_box[1 + fr * 6] * ctx->Full_box[4 + fr * 6]) /
              (ctx->Full_box[0 + fr * 6] * ctx->Full_box[3 + fr * 6] *
               ctx->Full_box[5 + fr * 6]) -
          ctx->Full_box[2 + fr * 6] /
              (ctx->Full_box[0 + fr * 6] * ctx->Full_box[5 + fr * 6]);
      ctx->nowinobox[3 + fr * 6] = real(1.) / ctx->Full_box[3 + fr * 6];
      ctx->nowinobox[4 + fr * 6] =
          -ctx->Full_box[4 + fr * 6] /
          (ctx->Full_box[3 + fr * 6] * ctx->Full_box[5 + fr * 6]);
      ctx->nowinobox[5 + fr * 6] = real(1.) / ctx->Full_box[5 + fr * 6];
    }
    cudaMemcpyAsync(ctx->nowinobox_d, ctx->nowinobox, sizeof(real) * nf * 6,
                    cudaMemcpyHostToDevice, stream);

    convert_carte_to_int_launcher(ctx->nowinobox_d, nowpos_d, ctx->nowinopos_d, N,
                                  nf, stream);

    ctx->MAX_PARTICLE_CELLS = N / 3;
    if (ctx->MAX_PARTICLE_CELLS < 1) ctx->MAX_PARTICLE_CELLS = 1;
    if (ctx->MAX_PARTICLE_CELLS > BLOCK_DIM_SAFE)
      ctx->MAX_PARTICLE_CELLS = BLOCK_DIM_SAFE;

    for (int fr = 0; fr < nf; fr++) {
      int c_nx, c_ny, c_nz;
      celleCompute(N, ctx->Full_box + fr * 6, ctx->nowinopos_d + fr * 3 * N, C.R_c,
                   &ctx->Cells, &ctx->Cells_howmany, &c_nx, &c_ny, &c_nz,
                   ctx->MAX_PARTICLE_CELLS, &ctx->cells_capacity_num,
                   &ctx->cells_capacity_mpc, stream);
      imeCompute(N, nowbox_d + fr * 6, ctx->nowinopos_d + fr * 3 * N, C.R_c,
                 ctx->Cells, ctx->Cells_howmany, c_nx, c_ny, c_nz,
                 ctx->with_d + fr * N * C.Radbuff, ctx->howmany_d + N * fr,
                 ctx->with_dist2_d + fr * N * C.Radbuff, ctx->MAX_PARTICLE_CELLS,
                 C.Radbuff, stream);
    }

    {
      int* howmany_h = (int*)malloc(sizeof(int) * nf * N);
      cudaMemcpyAsync(howmany_h, ctx->howmany_d, sizeof(int) * nf * N,
                      cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      for (int i = 0; i < nf * N; i++) {
        if (howmany_h[i] > C.Radbuff) {
          printf("Buffer radiale saturato by\n");
          printf("Particle slot %d with %d neighbours (Radbuff=%d)\n", i,
                 howmany_h[i], C.Radbuff);
          fflush(stdout);
          free(howmany_h);
          exit(0);
        }
      }
      free(howmany_h);
    }

    Tensor* raddescr_tensor = NULL;
    TensorShape raddescr_shape;
    raddescr_shape.AddDim(nf);
    raddescr_shape.AddDim(N);
    raddescr_shape.AddDim(C.Radbuff);
    OP_REQUIRES_OK(context, context->allocate_output(0, raddescr_shape,
                                                     &raddescr_tensor));

    Tensor* angdescr_tensor = NULL;
    TensorShape angdescr_shape;
    angdescr_shape.AddDim(nf);
    angdescr_shape.AddDim(N);
    angdescr_shape.AddDim(C.Angbuff);
    OP_REQUIRES_OK(context, context->allocate_output(1, angdescr_shape,
                                                     &angdescr_tensor));

    set_tensor_to_zero_real(angdescr_tensor->flat<real>().data(),
                            nf * N * C.Angbuff, stream);

    Tensor* des3bsupp_tensor = NULL;
    TensorShape des3bsupp_shape;
    des3bsupp_shape.AddDim(nf);
    des3bsupp_shape.AddDim(N);
    des3bsupp_shape.AddDim(C.Radbuff);
    OP_REQUIRES_OK(context, context->allocate_output(2, des3bsupp_shape,
                                                     &des3bsupp_tensor));

    Tensor* intmap2b_tensor = NULL;
    TensorShape intmap2b_shape;
    intmap2b_shape.AddDim(nf);
    intmap2b_shape.AddDim(N);
    intmap2b_shape.AddDim(C.Radbuff + 1);
    OP_REQUIRES_OK(context, context->allocate_output(3, intmap2b_shape,
                                                     &intmap2b_tensor));
    set_tensor_to_zero_int(intmap2b_tensor->flat<int>().data(),
                           nf * N * (C.Radbuff + 1), stream);

    Tensor* intmap3b_tensor = NULL;
    TensorShape intmap3b_shape;
    intmap3b_shape.AddDim(nf);
    intmap3b_shape.AddDim(N);
    intmap3b_shape.AddDim(C.Angbuff * 2);
    OP_REQUIRES_OK(context, context->allocate_output(4, intmap3b_shape,
                                                     &intmap3b_tensor));
    set_tensor_to_zero_int(intmap3b_tensor->flat<int>().data(),
                           nf * N * C.Angbuff * 2, stream);

    Tensor* der2b_tensor = NULL;
    TensorShape der2b_shape;
    der2b_shape.AddDim(nf);
    der2b_shape.AddDim(N);
    der2b_shape.AddDim(3);
    der2b_shape.AddDim(C.Radbuff);
    OP_REQUIRES_OK(context,
                   context->allocate_output(5, der2b_shape, &der2b_tensor));

    Tensor* der3b_tensor = NULL;
    TensorShape der3b_shape;
    der3b_shape.AddDim(nf);
    der3b_shape.AddDim(N);
    der3b_shape.AddDim(3);
    der3b_shape.AddDim(C.Angbuff * 2);
    OP_REQUIRES_OK(context,
                   context->allocate_output(6, der3b_shape, &der3b_tensor));

    Tensor* der3bsupp_tensor = NULL;
    TensorShape der3bsupp_shape;
    der3bsupp_shape.AddDim(nf);
    der3bsupp_shape.AddDim(N);
    der3bsupp_shape.AddDim(3);
    der3bsupp_shape.AddDim(C.Radbuff);
    OP_REQUIRES_OK(context, context->allocate_output(7, der3bsupp_shape,
                                                     &der3bsupp_tensor));

    Tensor* numtriplet_tensor = NULL;
    TensorShape numtriplet_shape;
    numtriplet_shape.AddDim(nf);
    numtriplet_shape.AddDim(N);
    OP_REQUIRES_OK(context, context->allocate_output(8, numtriplet_shape,
                                                     &numtriplet_tensor));

    set_tensor_to_zero_int(numtriplet_tensor->flat<int>().data(), nf * N, stream);

    real* rad_descr_d = raddescr_tensor->flat<real>().data();
    int* intmap2b_d = intmap2b_tensor->flat<int>().data();
    real* der2b_d = der2b_tensor->flat<real>().data();
    real* des3bsupp_d = des3bsupp_tensor->flat<real>().data();
    real* der3bsupp_d = der3bsupp_tensor->flat<real>().data();
    int* numtriplet_d = numtriplet_tensor->flat<int>().data();
    real* ang_descr_d = angdescr_tensor->flat<real>().data();
    int* intmap3b_d = intmap3b_tensor->flat<int>().data();
    real* der3b_d = der3b_tensor->flat<real>().data();

    fill_radial_launcher(C.R_c, C.Radbuff, C.R_a, C.Angbuff, N, ctx->nowinopos_d,
                         nowbox_d, ctx->howmany_d, ctx->with_d, rad_descr_d,
                         intmap2b_d, der2b_d, des3bsupp_d, der3bsupp_d, nf,
                         numtriplet_d, C.Rs, C.coeffA, C.coeffB, C.coeffC,
                         C.Pow_alpha, C.Pow_beta, stream);
    fill_angular_launcher(C.R_c, C.Radbuff, C.R_a, C.Angbuff, N, ctx->nowinopos_d,
                          nowbox_d, ctx->howmany_d, ctx->with_d, ang_descr_d,
                          intmap3b_d, des3bsupp_d, der3b_d, der3bsupp_d, nf,
                          numtriplet_d, stream);
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeDescriptorsLight").Device(DEVICE_GPU),
                        ComputeDescriptorsLightOp);
