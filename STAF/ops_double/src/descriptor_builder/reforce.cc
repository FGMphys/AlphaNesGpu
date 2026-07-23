#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <ctype.h>
#include "staf_real.h"

#include "vector.h"
#include "interaction_map.h"
#include "cell_list.h"
#include "smart_allocator.h"
#include "utilities.h"
#include <cuda_runtime.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#define PI real(3.141592654)
#define SQR(x) ((x)*(x))
#define Power(x,n) (staf_pow((real)(x),(real)(n)))

static int Radbuff,Angbuff;
static real R_c,Rs,R_a,coeffA,coeffB,coeffC,Pow_alpha,Pow_beta;

static real box[6],Inobox[6];
static vector* Nowinopos;
static interactionmap *Ime;
static listcell *Cells;

static real* Full_pos;
static real* Full_box;

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

void construct_descriptor(const real* box,int N,int max_batch){

          Inobox[0]=1./box[0];
          Inobox[1]=-box[1]/(box[0]*box[3]);
          Inobox[2]=(box[1]*box[4])/(box[0]*box[3]*box[5])-box[2]/(box[0]*box[5]);
          Inobox[3]=1./box[3];
          Inobox[4]=-box[4]/(box[3]*box[5]);
          Inobox[5]=1./box[5];

          Cells=getList(box,R_c,N);

          // INTERACTION MAPS
          Ime=createInteractionMap(N,Radbuff);
          //Memory for reticular positions
          Nowinopos=(vector*)calloc(N,sizeof(vector));
	  //Memory to copy input on CPU
	  int nf=max_batch;
	  Full_pos=(real*)calloc(nf*N*3,sizeof(real));
	  Full_box=(real*)calloc(nf*6,sizeof(real));
          
	  cudaMalloc(&howmany_d,nf*N*sizeof(int));
          cudaMalloc(&with_d,nf*N*Radbuff*sizeof(int));
          cudaMalloc(&nowinopos_d,nf*N*3*sizeof(real));
          cudaMalloc(&code_ret_d,sizeof(int));
	  code_ret=(int*)calloc(1,sizeof(int));
 }

 void fill_radial_launcher(real R_c,int radbuff,real R_a,int angbuff,int N,
                       real* inopos_d,const real* box_d,
                       int *howmany_d,int *with_d,
                       real* descriptor_d,int* intmap2b_d,real* der2b_d,
                       real* des3bsupp_d,
                       real* der3bsupp_d, int nf,int* numtriplet_d,
                       real rs, real coeffa,real coeffb,real coeffc,real pow_alpha, real pow_beta);
 void fill_angular_launcher(real R_c,int radbuff,real R_a,int angbuff,int N,
                       real* inopos_d,const real* box_d,
                       int *howmany_d,int *with_d,
                       real* ang_descr_d,int* intmap3b_d,
                       real* des3bsupp_d,real* der3b_d,
                       real* der3bsupp_d, int nf,int* numtriplet_d);

void set_tensor_to_zero_int(int* tensor,int dimten);

void set_tensor_to_zero_double(real* tensor,int dimten);

void check_max_launcher(int* tensor,int dim,int maxval,int* resval);

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
    // Grab the input tensor
    const Tensor& positions_T = context->input(0);
    const Tensor& box_T = context->input(1);


    auto positions = positions_T.flat<real>();
    const real* nowpos=positions.data();
    const real* nowbox = box_T.flat<real>().data();



    //Copio i tensori input in nuovi array per elaborarli
    int nf=box_T.shape().dim_size(0);
    int N=int(positions_T.shape().dim_size(1)/3);

    cudaMemcpy(Full_pos,nowpos,nf*N*3*sizeof(real),cudaMemcpyDeviceToHost);
    cudaMemcpy(Full_box,nowbox,nf*6*sizeof(real),cudaMemcpyDeviceToHost);
    //////////BUILDING CELL LIST AND IME (FULL ORDERED INTERACTION MAP)////
    int ii;
    for (ii=0;ii<nf;ii++)
    {
      Inobox[0] = real(1.) / Full_box[ii*6+0];
      Inobox[1] = -Full_box[ii*6+1] / (Full_box[ii*6+0] * Full_box[ii*6+3]);
      Inobox[2] = (Full_box[ii*6+1] * Full_box[ii*6+4]) /
                  (Full_box[ii*6+0] * Full_box[ii*6+3] * Full_box[ii*6+5]) -
                  Full_box[ii*6+2] / (Full_box[ii*6+0] * Full_box[ii*6+5]);
      Inobox[3] = real(1.) / Full_box[ii*6+3];
      Inobox[4] = -Full_box[ii*6+4] / (Full_box[ii*6+3] * Full_box[ii*6+5]);
      Inobox[5] = real(1.) / Full_box[ii*6+5];

      for (int i=0;i<N;i++){
        real px=Full_pos[ii*N*3+i*3];
        real py=Full_pos[ii*N*3+i*3+1];
        real pz=Full_pos[ii*N*3+i*3+2];

        Nowinopos[i].x=(Inobox[0]*px+Inobox[1]*py+Inobox[2]*pz);
        Nowinopos[i].y=(Inobox[3]*py+Inobox[4]*pz);
        Nowinopos[i].z=(Inobox[5]*pz);
      }

      // calcolo delle celle e dei neighbour list
      fullUpdateList(Cells,Nowinopos,N,&Full_box[ii*6],R_c);
      resetInteractionMap(Ime);
      calculateInteractionMapWithCutoffDistanceOrdered(Cells,Ime,Nowinopos,&Full_box[ii*6],R_c);

      cudaMemcpy(howmany_d+ii*N,Ime->howmany,N*sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(with_d+ii*N*Radbuff,Ime->with[0],N*Radbuff*sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(nowinopos_d+ii*N*3,Nowinopos,N*3*sizeof(real),cudaMemcpyHostToDevice);

      for (int i=0;i<N;i++)
      {
        if (Ime->howmany[i]>Radbuff)
        {
          printf("Buffer radiale saturato by \n");
	  printf("Particle %d at frame %d with %d neighbours \n",i,ii,Ime->howmany[i]);
          fflush(stdout);
	  exit(0);
        }
      }

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

    set_tensor_to_zero_double(angdescr_tensor->flat<real>().data(),nf*N*Angbuff);
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
    set_tensor_to_zero_int(intmap2b_tensor->flat<int>().data(),nf*N*(Radbuff+1));
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
    set_tensor_to_zero_int(intmap3b_tensor->flat<int>().data(),nf*N*Angbuff*2);
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

    set_tensor_to_zero_int(numtriplet_tensor->flat<int>().data(),nf*N);

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
                      nowinopos_d,nowbox,
                      howmany_d,with_d,
                      rad_descr_d,intmap2b_d,der2b_d,
                      des3bsupp_d,
                      der3bsupp_d,nf,numtriplet_d,
                      Rs,coeffA,coeffB,coeffC,Pow_alpha,Pow_beta);
    //cudaMemset(code_ret_d,sizeof(int),0);
    //check_max_launcher(numtriplet_d,N*nf,Angbuff,code_ret_d);
    //cudaMemcpy(code_ret,code_ret_d,sizeof(int),cudaMemcpyDeviceToHost);
    //if (code_ret[0]!=0){
    //   printf("alpha_nes: Buffer angolare saturato, %d vs %d",code_ret[0],Angbuff);
    //   exit(0);
    // }
    fill_angular_launcher(R_c, Radbuff, R_a, Angbuff, N, nowinopos_d,
		         nowbox, howmany_d, with_d, ang_descr_d,
			 intmap3b_d, des3bsupp_d, der3b_d, der3bsupp_d,
			 nf, numtriplet_d);
     }




   };
   REGISTER_KERNEL_BUILDER(Name("ComputeDescriptorsLight").Device(DEVICE_GPU), ComputeDescriptorsLightOp);
