#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <ctype.h>

#include "vector.h"
#include "interaction_map.h"
#include "cell_list.h"
#include "smart_allocator.h"
#include "utilities.h"
#include <cuda_runtime.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "staf_real.h"

#define PI real(3.141592654)
#define SQR(x) ((x)*(x))
#define Power(x,n) (staf_pow(x,n))

static int Radbuff,Angbuff;
static real R_c,Rs,R_a,Pow_alpha,Pow_beta;
static real Rs_inter,Rc_inter,Ra_inter;

static real Rc_celle;
static real coeffA_intra,coeffB_intra,coeffC_intra,coeffA_inter,coeffB_inter,coeffC_inter;

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



void save_cutoff_intra(real rc){
  FILE *newfile;
  newfile=fopen("cutoff_curve_intra.dat","w");
  real dx=rc/1000.;
  real x=0;
  for (int k=0;k<1000;k++){
    x=x+dx;
    if (x<Rs){
      fprintf(newfile,"%g %g\n",x,coeffA_intra/Power(x/Rs,Pow_alpha)+coeffB_intra/Power(x/Rs,Pow_beta)+coeffC_intra);
    }
    else{
      fprintf(newfile,"%g %g\n",x,0.5*(1+staf_cos(PI*x/rc)));
  }
}
   fclose(newfile);
}


void save_cutoff_inter(real rc){
  FILE *newfile;
  newfile=fopen("cutoff_curve_inter.dat","w");
  real dx=rc/1000.;
  real x=0;
  for (int k=0;k<1000;k++){
    x=x+dx;
    if (x<Rs_inter){
      fprintf(newfile,"%g %g\n",x,coeffA_inter/Power(x/Rs_inter,Pow_alpha)+coeffB_inter/Power(x/Rs_inter,Pow_beta)+coeffC_inter);
    }
    else{
      fprintf(newfile,"%g %g\n",x,0.5*(1+staf_cos(PI*x/rc)));
  }
}
   fclose(newfile);
}

void construct_repulsion_intra(){
    real alpha=1.;
    real beta=-30.;
    Pow_alpha=alpha;
    Pow_beta=beta;
    real rs=Rs;
    real rc=R_c;
    real f=0.5*(staf_cos(PI*rs/rc)+1);
    real f1=-0.5*PI/rc*staf_sin(PI*rs/rc);
    real f2=-0.5*SQR(PI/rc)*staf_cos(PI*rs/rc);

    coeffB_intra=(f1*rs+f2*SQR(rs)/(alpha+1))*(alpha+1)/beta/(beta-alpha);
    coeffA_intra=(f2*SQR(rs)-coeffB_intra*beta*(beta+1))/(alpha*(alpha+1));
    coeffC_intra=f-coeffA_intra-coeffB_intra;

    save_cutoff_intra(rc);

}



void construct_repulsion_inter(){
    real alpha=1.;
    real beta=-30.;
    Pow_alpha=alpha;
    Pow_beta=beta;
    real rs=Rs_inter;
    real rc=Rc_inter;
    real f=0.5*(staf_cos(PI*rs/rc)+1);
    real f1=-0.5*PI/rc*staf_sin(PI*rs/rc);
    real f2=-0.5*SQR(PI/rc)*staf_cos(PI*rs/rc);

    coeffB_inter=(f1*rs+f2*SQR(rs)/(alpha+1))*(alpha+1)/beta/(beta-alpha);
    coeffA_inter=(f2*SQR(rs)-coeffB_inter*beta*(beta+1))/(alpha*(alpha+1));
    coeffC_inter=f-coeffA_inter-coeffB_inter;

    save_cutoff_inter(rc);

}

void construct_descriptor(const real* box,int N,int max_batch){

          Inobox[0]=1./box[0];
          Inobox[1]=-box[1]/(box[0]*box[3]);
          Inobox[2]=(box[1]*box[4])/(box[0]*box[3]*box[5])-box[2]/(box[0]*box[5]);
          Inobox[3]=1./box[3];
          Inobox[4]=-box[4]/(box[3]*box[5]);
          Inobox[5]=1./box[5];

          Cells=getList(box,Rc_celle,N);

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
                       real rs, real coeffa_intra,real coeffb_intra,real coeffc_intra,
                       real coeffa_inter,real coeffb_inter,real coeffc_inter,
                       real pow_alpha, real pow_beta,
                      real Rs_inter,real Rc_inter,const int* map_intra_d,real Ra_inter);
 void fill_angular_launcher(real R_c,int radbuff,real R_a,int angbuff,int N,
                       real* inopos_d,const real* box_d,
                       int *howmany_d,int *with_d,
                       real* ang_descr_d,int* intmap3b_d,
                       real* des3bsupp_d,real* der3b_d,
                       real* der3bsupp_d, int nf,int* numtriplet_d,
                       const int* map_intra_d,real Ra_inter);

void set_tensor_to_zero_int(int* tensor,int dimten);

void set_tensor_to_zero_real(real* tensor,int dimten);

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
    .Input("rs_inter: " STAF_TF_DTYPE)
    .Input("rc_inter: " STAF_TF_DTYPE)
    .Input("ra_inter: " STAF_TF_DTYPE)
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

           const Tensor& rs_inter_T = context->input(8);
           const Tensor& rc_inter_T = context->input(9);
           const Tensor& ra_inter_T = context->input(10);

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

           auto Rs_inter_flat = rs_inter_T.flat<real>();
           auto Rc_inter_flat = rc_inter_T.flat<real>();
           auto Ra_inter_flat = ra_inter_T.flat<real>();
          
           if (Rc_inter>R_c){
              Rc_celle=Rc_inter;
              printf("\nAlpha_nes: Found intra-range < inter-range, cell cut-off is %lf \n",Rc_celle);
           }
            else{
              Rc_celle=R_c;
              printf("\nAlpha_nes: Found intra-range > inter-range, cell cut-off is %lf \n",Rc_celle);
            }

           Rs_inter= Rs_inter_flat(0);
           Rc_inter = Rc_inter_flat(0);
           Ra_inter=Ra_inter_flat(0);

	         int numpar=numpar_T.flat<int>()(0);
           int max_batch=max_batch_T.flat<int>()(0);
           printf("\nAlpha_nes: Descriptor constructor found Rc_intra %f\n",R_c);
	   printf("          Rs_intra %f Ra_intra %f Rc_inter %f Rs_inter %f Ra_inter %f\n",Rs,R_a,Rc_inter,Rs_inter,Ra_inter);
     printf("Radbuff %d Angbuff %d max_batch %d N_max %d\n",Radbuff,Angbuff,max_batch,numpar);
           construct_repulsion_intra();
	   construct_repulsion_inter();
           construct_descriptor(box_T.flat<real>().data(),numpar,max_batch);
         }
    };
REGISTER_KERNEL_BUILDER(Name("ConstructDescriptorsLight").Device(DEVICE_CPU), ConstructDescriptorsLightOp);


REGISTER_OP("ComputeDescriptorsLight")
    .Input("positions: " STAF_TF_DTYPE)
    .Input("boxer: " STAF_TF_DTYPE)
    .Input("map_intra: int32")
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
    const Tensor& map_intra_T = context->input(2);
    auto map_intra_d_flat = map_intra_T.flat<int>();


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
      fullUpdateList(Cells,Nowinopos,N,&Full_box[ii*6],Rc_celle);
      resetInteractionMap(Ime);
      calculateInteractionMapWithCutoffDistanceOrdered(Cells,Ime,Nowinopos,&Full_box[ii*6],Rc_celle);

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

    set_tensor_to_zero_real(raddescr_tensor->flat<real>().data(),nf*N*Radbuff);
    // Create an output tensor
    Tensor* angdescr_tensor = NULL;
    TensorShape angdescr_shape;
    angdescr_shape.AddDim (nf);
    angdescr_shape.AddDim (N);
    angdescr_shape.AddDim (Angbuff);
    OP_REQUIRES_OK(context, context->allocate_output(1,angdescr_shape,
                                                     &angdescr_tensor));

    set_tensor_to_zero_real(angdescr_tensor->flat<real>().data(),nf*N*Angbuff);
    ///////////////DESCRIPTORS 3B SUPP///////////////
    // Create an output tensor
    Tensor* des3bsupp_tensor = NULL;
    TensorShape des3bsupp_shape;
    des3bsupp_shape.AddDim (nf);
    des3bsupp_shape.AddDim (N);
    des3bsupp_shape.AddDim (Radbuff);
    OP_REQUIRES_OK(context, context->allocate_output(2,des3bsupp_shape,
                                                     &des3bsupp_tensor));
    set_tensor_to_zero_real(des3bsupp_tensor->flat<real>().data(),nf*N*Radbuff);
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

    set_tensor_to_zero_real(der2b_tensor->flat<real>().data(),nf*N*3*Radbuff);
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

    set_tensor_to_zero_real(der3b_tensor->flat<real>().data(),nf*N*3*2*Angbuff);
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
    set_tensor_to_zero_real(der3bsupp_tensor->flat<real>().data(),nf*N*3*Radbuff);
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
                      Rs,coeffA_intra,coeffB_intra,coeffC_intra,
                      coeffA_inter,coeffB_inter,coeffC_inter,
                      Pow_alpha,Pow_beta,Rs_inter,Rc_inter,
                      map_intra_d_flat.data(),Ra_inter);
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
			 nf, numtriplet_d,map_intra_d_flat.data(),Ra_inter);
    
 
  }




   };
   REGISTER_KERNEL_BUILDER(Name("ComputeDescriptorsLight").Device(DEVICE_GPU), ComputeDescriptorsLightOp);
