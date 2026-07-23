#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <ctype.h>
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

static int Radbuff,Angbuff;
static real R_c,Rs,R_a,coeffA,coeffB,coeffC,Pow_alpha,Pow_beta;

static real* Full_box;
static real* nowinobox;
static real* nowinobox_d;
static real* with_dist2_d;

static int *Cells;
static int *Cells_howmany;
static int cells_capacity_num;
static int cells_capacity_mpc;
static int MAX_PARTICLE_CELLS;

static int *howmany_d;
static int *with_d;
static int *code_ret_d;
static int *code_ret;
static real *nowinopos_d;



void save_cutoff(real rc){
  FILE *newfile;
  newfile=fopen("cutoff_curve.dat","w");
  real dx=rc/1000.;
  real x=0;
  for (int k=0;k<1000;k++){
    x=x+dx;
    if (x<Rs){
      fprintf(newfile,"%g %g\n",x,coeffA/Power(x,Pow_alpha)+coeffB/Power(x,Pow_beta)+coeffC);
    }
    else{
      fprintf(newfile,"%g %g\n",x,0.5*(1+staf_cos(PI*x/rc)));
  }
}
   fclose(newfile);
}
void construct_repulsion(){
    real alpha=1.;
    real beta=-30.;
    Pow_alpha=alpha;
    Pow_beta=beta;
    real rs=Rs;
    real rc=R_c;
    real f=0.5*(staf_cos(PI*rs/rc)+1);
    real f1=-0.5*PI/rc*staf_sin(PI*rs/rc);
    real f2_red=-0.5*SQR(PI/rc)*staf_cos(PI*rs/rc)*SQR(rs);
    real gamma_red=1./(alpha-beta)*alpha-1;
    real delta_red=1./(alpha-beta)*(f*(alpha-beta)-f1*rs-f*alpha);
    real eta_red=-alpha/(alpha-beta);
    real epsilon_red=1./(alpha-beta)*(rs*f1+alpha*f);
    real c2_red=alpha*(alpha+1)*delta_red+beta*(beta+1)*epsilon_red;
    real c1_red=alpha*(alpha+1)*gamma_red+beta*(beta+1)*eta_red;
    coeffC=(f2_red-c2_red)/c1_red;
    real eta=-alpha*Power(rs,beta)/(alpha-beta);
    real epsilon=Power(rs,beta)/(alpha-beta)*(rs*f1+alpha*f);
    coeffB=eta*coeffC+epsilon;
    real gamma=Power(rs,alpha)/(alpha-beta)*alpha-Power(rs,alpha);
    real delta=Power(rs,alpha)/(alpha-beta)*(f*(alpha-beta)-f1*rs-f*alpha);
    coeffA=gamma*coeffC+delta;
    save_cutoff(rc);

}

void construct_descriptor(const real* /*box*/,int N,int max_batch){
          MAX_PARTICLE_CELLS=N/3;
          if (MAX_PARTICLE_CELLS < 1) MAX_PARTICLE_CELLS = 1;
          if (MAX_PARTICLE_CELLS > BLOCK_DIM_SAFE) MAX_PARTICLE_CELLS = BLOCK_DIM_SAFE;
          int nf=max_batch;
          Full_box=(real*)calloc(nf*6,sizeof(real));
          nowinobox=(real*)calloc(nf*6,sizeof(real));
          cudaMalloc(&nowinobox_d,nf*6*sizeof(real));

          cudaMalloc(&with_dist2_d,nf*N*Radbuff*sizeof(real));
          cudaMalloc(&howmany_d,nf*N*sizeof(int));
          cudaMalloc(&with_d,nf*N*Radbuff*sizeof(int));
          cudaMalloc(&nowinopos_d,nf*N*3*sizeof(real));
          cudaMalloc(&code_ret_d,sizeof(int));
          code_ret=(int*)calloc(1,sizeof(int));

          Cells=nullptr;
          Cells_howmany=nullptr;
          cells_capacity_num=0;
          cells_capacity_mpc=0;
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

           auto rs_T_flat=rs_T.flat<real>();
           Rs=rs_T_flat(0);

           auto rcrad_T_flat=rcrad_T.flat<real>();
           R_c=rcrad_T_flat(0);

           auto radbuff_T_flat=radbuff_T.flat<int>();
           Radbuff=radbuff_T_flat(0);

           auto angbuff_T_flat=angbuff_T.flat<int>();
           Angbuff=angbuff_T_flat(0);

           auto ra_T_flat=ra_T.flat<real>();
           R_a=ra_T_flat(0);

	   int numpar=numpar_T.flat<int>()(0);
           int max_batch=max_batch_T.flat<int>()(0);
           printf("\nSTAF: Descriptor constructor found Rc %f\n",R_c);
	   printf("          Ra %f Rs %f Radbuff %d Angbuff %d max_batch %d N_max %d\n",R_a,Rs,Radbuff,Angbuff,max_batch,numpar);
           construct_repulsion();
           construct_descriptor(box_T.flat<real>().data(),numpar,max_batch);
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

    // Grab the input tensor
    const Tensor& positions_T = context->input(0);
    const Tensor& box_T = context->input(1);


    auto positions = positions_T.flat<real>();
    const real* nowpos_d=positions.data();
    const real* nowbox_d = box_T.flat<real>().data();



    //Copio i tensori input in nuovi array per elaborarli
    int nf=box_T.shape().dim_size(0);
    int N=int(positions_T.shape().dim_size(1)/3);

    // Host box only (tiny): cell grid sizing. Positions stay on GPU.
    cudaMemcpyAsync(Full_box, nowbox_d, sizeof(real)*nf*6,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int fr=0; fr<nf; fr++){
      nowinobox[0+fr*6]=real(1.)/Full_box[0+fr*6];
      nowinobox[1+fr*6]=-Full_box[1+fr*6]/(Full_box[0+fr*6]*Full_box[3+fr*6]);
      nowinobox[2+fr*6]=(Full_box[1+fr*6]*Full_box[4+fr*6])/(Full_box[0+fr*6]*Full_box[3+fr*6]*Full_box[5+fr*6])-Full_box[2+fr*6]/(Full_box[0+fr*6]*Full_box[5+fr*6]);
      nowinobox[3+fr*6]=real(1.)/Full_box[3+fr*6];
      nowinobox[4+fr*6]=-Full_box[4+fr*6]/(Full_box[3+fr*6]*Full_box[5+fr*6]);
      nowinobox[5+fr*6]=real(1.)/Full_box[5+fr*6];
    }
    cudaMemcpyAsync(nowinobox_d, nowinobox, sizeof(real)*nf*6,
                    cudaMemcpyHostToDevice, stream);

    convert_carte_to_int_launcher(nowinobox_d, nowpos_d, nowinopos_d, N, nf,
                                  stream);

    MAX_PARTICLE_CELLS=N/3;
    if (MAX_PARTICLE_CELLS < 1) MAX_PARTICLE_CELLS = 1;
    if (MAX_PARTICLE_CELLS > BLOCK_DIM_SAFE) MAX_PARTICLE_CELLS = BLOCK_DIM_SAFE;

    for (int fr=0; fr<nf; fr++){
      int c_nx,c_ny,c_nz;
      celleCompute(N, Full_box+fr*6, nowinopos_d+fr*3*N, R_c,
                   &Cells, &Cells_howmany, &c_nx, &c_ny, &c_nz,
                   MAX_PARTICLE_CELLS, &cells_capacity_num, &cells_capacity_mpc,
                   stream);
      imeCompute(N, nowbox_d+fr*6, nowinopos_d+fr*3*N, R_c,
                 Cells, Cells_howmany, c_nx, c_ny, c_nz,
                 with_d+fr*N*Radbuff, howmany_d+N*fr,
                 with_dist2_d+fr*N*Radbuff, MAX_PARTICLE_CELLS, Radbuff,
                 stream);
    }

    // Overflow check: howmany > Radbuff
    cudaMemsetAsync(code_ret_d, 0, sizeof(int), stream);
    {
      // Reuse check_max only for triplets; do a compact host scan of howmany.
      int* howmany_h = (int*)malloc(sizeof(int)*nf*N);
      cudaMemcpyAsync(howmany_h, howmany_d, sizeof(int)*nf*N,
                      cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      for (int i=0;i<nf*N;i++){
        if (howmany_h[i] > Radbuff){
          printf("Buffer radiale saturato by\n");
          printf("Particle slot %d with %d neighbours (Radbuff=%d)\n",
                 i, howmany_h[i], Radbuff);
          fflush(stdout);
          free(howmany_h);
          exit(0);
        }
      }
      free(howmany_h);
    }

    ///////////////DESCRIPTORS///////////////
    // Create an output tensor
    Tensor* raddescr_tensor = NULL;
    TensorShape raddescr_shape;
    raddescr_shape.AddDim (nf);
    raddescr_shape.AddDim (N);
    raddescr_shape.AddDim (Radbuff);
    OP_REQUIRES_OK(context, context->allocate_output(0,raddescr_shape,
                                                     &raddescr_tensor));

    // Create an output tensor
    Tensor* angdescr_tensor = NULL;
    TensorShape angdescr_shape;
    angdescr_shape.AddDim (nf);
    angdescr_shape.AddDim (N);
    angdescr_shape.AddDim (Angbuff);
    OP_REQUIRES_OK(context, context->allocate_output(1,angdescr_shape,
                                                     &angdescr_tensor));

    set_tensor_to_zero_real(angdescr_tensor->flat<real>().data(),nf*N*Angbuff, stream);
    ///////////////DESCRIPTORS 3B SUPP///////////////
    // Create an output tensor
    Tensor* des3bsupp_tensor = NULL;
    TensorShape des3bsupp_shape;
    des3bsupp_shape.AddDim (nf);
    des3bsupp_shape.AddDim (N);
    des3bsupp_shape.AddDim (Radbuff);
    OP_REQUIRES_OK(context, context->allocate_output(2,des3bsupp_shape,
                                                     &des3bsupp_tensor));
    ///////////////INTMAP2B///////////////
    // Create an output tensor
    Tensor* intmap2b_tensor = NULL;
    TensorShape intmap2b_shape;
    intmap2b_shape.AddDim (nf);
    intmap2b_shape.AddDim (N);
    intmap2b_shape.AddDim (Radbuff+1);
    OP_REQUIRES_OK(context, context->allocate_output(3,intmap2b_shape,
                                                     &intmap2b_tensor));
    set_tensor_to_zero_int(intmap2b_tensor->flat<int>().data(),nf*N*(Radbuff+1), stream);
    /////////////////////////////
    ///////////////INTMAP3B///////////////
    // Create an output tensor
    Tensor* intmap3b_tensor = NULL;
    TensorShape intmap3b_shape;
    intmap3b_shape.AddDim (nf);
    intmap3b_shape.AddDim (N);
    intmap3b_shape.AddDim (Angbuff*2);
    OP_REQUIRES_OK(context, context->allocate_output(4,intmap3b_shape,
                                                     &intmap3b_tensor));
    set_tensor_to_zero_int(intmap3b_tensor->flat<int>().data(),nf*N*Angbuff*2, stream);
    /////////////////////////////
    ///////////////DER2B///////////////
    // Create an output tensor
    Tensor* der2b_tensor = NULL;
    TensorShape der2b_shape;
    der2b_shape.AddDim (nf);
    der2b_shape.AddDim (N);
    der2b_shape.AddDim (3);
    der2b_shape.AddDim (Radbuff);
    OP_REQUIRES_OK(context, context->allocate_output(5,der2b_shape,
                                                     &der2b_tensor));
    /////////////////////////////
    ///////////////DER3B///////////////
    // Create an output tensor
    Tensor* der3b_tensor = NULL;
    TensorShape der3b_shape;
    der3b_shape.AddDim (nf);
    der3b_shape.AddDim (N);
    der3b_shape.AddDim (3);
    der3b_shape.AddDim (Angbuff*2);
    OP_REQUIRES_OK(context, context->allocate_output(6,der3b_shape,
                                                     &der3b_tensor));


    /////////////////////////////
    ///////////////DER3B_SUPP///////////////
    // Create an output tensor
    Tensor* der3bsupp_tensor = NULL;
    TensorShape der3bsupp_shape;
    der3bsupp_shape.AddDim (nf);
    der3bsupp_shape.AddDim (N);
    der3bsupp_shape.AddDim (3);
    der3bsupp_shape.AddDim (Radbuff);
    OP_REQUIRES_OK(context, context->allocate_output(7,der3bsupp_shape,
                                                     &der3bsupp_tensor));

    // Create an output tensor
    Tensor* numtriplet_tensor = NULL;
    TensorShape numtriplet_shape;
    numtriplet_shape.AddDim (nf);
    numtriplet_shape.AddDim (N);
    OP_REQUIRES_OK(context, context->allocate_output(8,numtriplet_shape,
                                                     &numtriplet_tensor));

    set_tensor_to_zero_int(numtriplet_tensor->flat<int>().data(),nf*N, stream);

    real* rad_descr_d=raddescr_tensor->flat<real>().data();
    int* intmap2b_d=intmap2b_tensor->flat<int>().data();
    real* der2b_d=der2b_tensor->flat<real>().data();
    real* des3bsupp_d=des3bsupp_tensor->flat<real>().data();
    real* der3bsupp_d=der3bsupp_tensor->flat<real>().data();
    int* numtriplet_d=numtriplet_tensor->flat<int>().data();

    real* ang_descr_d=angdescr_tensor->flat<real>().data();
    int* intmap3b_d=intmap3b_tensor->flat<int>().data();
    real* der3b_d=der3b_tensor->flat<real>().data();

    fill_radial_launcher(R_c,Radbuff,R_a,Angbuff,N,
                      nowinopos_d,nowbox_d,
                      howmany_d,with_d,
                      rad_descr_d,intmap2b_d,der2b_d,
                      des3bsupp_d,
                      der3bsupp_d,nf,numtriplet_d,
                      Rs,coeffA,coeffB,coeffC,Pow_alpha,Pow_beta, stream);
    fill_angular_launcher(R_c, Radbuff, R_a, Angbuff, N, nowinopos_d,
		         nowbox_d, howmany_d, with_d, ang_descr_d,
			 intmap3b_d, des3bsupp_d, der3b_d, der3bsupp_d,
			 nf, numtriplet_d, stream);
     }




   };
   REGISTER_KERNEL_BUILDER(Name("ComputeDescriptorsLight").Device(DEVICE_GPU), ComputeDescriptorsLightOp);
