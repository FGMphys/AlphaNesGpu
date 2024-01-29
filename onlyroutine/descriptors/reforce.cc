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
//#include "log.h"
#include <cuda_runtime.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#define PI 3.141592654f
#define SQR(x) ((x)*(x))
#define Power(x,n) (pow(x,n))

static int Radbuff,Angbuff;
static float R_c,Rs,R_a,coeffA,coeffB,coeffC,Pow_alpha,Pow_beta;

static float box[6],Inobox[6];
static vector* Nowinopos;
static interactionmap *Ime;
static listcell *Cells;

static float* Full_pos;
static float* Full_box;

static int *howmany_d;
static int *with_d;
static float *nowinopos_d;



void save_cutoff(float rc){
  FILE *newfile;
  newfile=fopen("cutoff_curve.dat","w");
  float dx=rc/1000.;
  float x=0;
  for (int k=0;k<1000;k++){
    x=x+dx;
    if (x<Rs){
      fprintf(newfile,"%g %g\n",x,coeffA/Power(x,Pow_alpha)+coeffB/Power(x,Pow_beta)+coeffC);
    }
    else{
      fprintf(newfile,"%g %g\n",x,0.5*(1+cos(PI*x/rc)));
  }
}
   fclose(newfile);
}
void construct_repulsion(){
    float alpha=1.;
    float beta=-30.;
    Pow_alpha=alpha;
    Pow_beta=beta;
    float rs=Rs;
    float rc=R_c;
    float f=0.5*(cos(PI*rs/rc)+1);
    float f1=-0.5*PI/rc*sin(PI*rs/rc);
    float f2_red=-0.5*SQR(PI/rc)*cos(PI*rs/rc)*SQR(rs);
    float gamma_red=1./(alpha-beta)*alpha-Power(rs,alpha);
    float delta_red=1./(alpha-beta)*(f*(alpha-beta)-f1*rs-f*alpha);
    float eta_red=-alpha/(alpha-beta);
    float epsilon_red=1./(alpha-beta)*(rs*f1+alpha*f);
    float c2_red=alpha*(alpha+1)*delta_red+beta*(beta+1)*epsilon_red;
    float c1_red=alpha*(alpha+1)*gamma_red+beta*(beta+1)*eta_red;
    coeffC=(f2_red-c2_red)/c1_red;
    float eta=-alpha*Power(rs,beta)/(alpha-beta);
    float epsilon=Power(rs,beta)/(alpha-beta)*(rs*f1+alpha*f);
    coeffB=eta*coeffC+epsilon;
    float gamma=Power(rs,alpha)/(alpha-beta)*alpha-Power(rs,alpha);
    float delta=Power(rs,alpha)/(alpha-beta)*(f*(alpha-beta)-f1*rs-f*alpha);
    coeffA=gamma*coeffC+delta;
    save_cutoff(rc);

}

void construct_descriptor(const float* box,int N,int max_batch){

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
	  Full_pos=(float*)calloc(nf*N*3,sizeof(float));
	  Full_box=(float*)calloc(nf*6,sizeof(float));
          
	  cudaMalloc(&howmany_d,nf*N*sizeof(int));
          cudaMalloc(&with_d,nf*N*Radbuff*sizeof(int));
          cudaMalloc(&nowinopos_d,nf*N*3*sizeof(float));
 }

 void fill_radial_launcher(float R_c,int radbuff,float R_a,int angbuff,int N,
                       float* inopos_d,const float* box_d,
                       int *howmany_d,int *with_d,
                       float* descriptor_d,int* intmap2b_d,float* der2b_d,
                       float* des3bsupp_d,
                       float* der3bsupp_d, int nf,int* numtriplet_d,
                       float rs, float coeffa,float coeffb,float coeffc,float pow_alpha, float pow_beta);
 void fill_angular_launcher(float R_c,int radbuff,float R_a,int angbuff,int N,
                       float* inopos_d,const float* box_d,
                       int *howmany_d,int *with_d,
                       float* ang_descr_d,int* intmap3b_d,
                       float* des3bsupp_d,float* der3b_d,
                       float* der3bsupp_d, int nf,int* numtriplet_d);

using namespace tensorflow;

REGISTER_OP("ConstructDescriptorsLight")
    .Input("radial_cutoff: float")
    .Input("radial_buffer: int32")
    .Input("angular_buffer: int32")
    .Input("numpar: int32")
    .Input("boxer: float")
    .Input("rs: float")
    .Input("ra: float")
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

           auto rs_T_flat=rs_T.flat<float>();
           Rs=rs_T_flat(0);

           auto rcrad_T_flat=rcrad_T.flat<float>();
           R_c=rcrad_T_flat(0);

           auto radbuff_T_flat=radbuff_T.flat<int>();
           Radbuff=radbuff_T_flat(0);

           auto angbuff_T_flat=angbuff_T.flat<int>();
           Angbuff=angbuff_T_flat(0);

           auto ra_T_flat=ra_T.flat<float>();
           R_a=ra_T_flat(0);

	   int numpar=numpar_T.flat<int>()(0);
           int max_batch=max_batch_T.flat<int>()(0);
           printf("\nAlpha_nes: Descriptor constructor found Rc %f\n",R_c);
	   printf("          Ra %f Rs %f Radbuff %d Angbuff %d max_batch %d\n",R_a,Rs,Radbuff,Angbuff,max_batch);
           construct_repulsion();
           construct_descriptor(box_T.flat<float>().data(),numpar,max_batch);
         }
    };
REGISTER_KERNEL_BUILDER(Name("ConstructDescriptorsLight").Device(DEVICE_CPU), ConstructDescriptorsLightOp);


REGISTER_OP("ComputeDescriptorsLight")
    .Input("positions: float")
    .Input("boxer: float")
    .Output("raddescr: float")
    .Output("angdescr: float")
    .Output("des3bsupp: float")
    .Output("intmap2b: int32")
    .Output("intmap3b: int32")
    .Output("der2b: float")
    .Output("der3b: float")
    .Output("der3bsupp: float")
    .Output("numtriplet: int32");


class ComputeDescriptorsLightOp : public OpKernel {
 public:
  explicit ComputeDescriptorsLightOp(OpKernelConstruction* context) : OpKernel(context) {


  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& positions_T = context->input(0);
    const Tensor& box_T = context->input(1);


    auto positions = positions_T.flat<float>();
    const float* nowpos=positions.data();
    const float* nowbox = box_T.flat<float>().data();



    //Copio i tensori input in nuovi array per elaborarli
    int nf=box_T.shape().dim_size(0);
    int N=int(positions_T.shape().dim_size(1)/3);

    cudaMemcpy(Full_pos,nowpos,nf*N*3*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(Full_box,nowbox,nf*6*sizeof(float),cudaMemcpyDeviceToHost);
    //////////BUILDING CELL LIST AND IME (FULL ORDERED INTERACTION MAP)////
    int ii;
    for (ii=0;ii<nf;ii++)
    {
      Inobox[0] = 1.f / Full_box[ii*6+0];
      Inobox[1] = -Full_box[ii*6+1] / (Full_box[ii*6+0] * Full_box[ii*6+3]);
      Inobox[2] = (Full_box[ii*6+1] * Full_box[ii*6+4]) /
                  (Full_box[ii*6+0] * Full_box[ii*6+3] * Full_box[ii*6+5]) -
                  Full_box[ii*6+2] / (Full_box[ii*6+0] * Full_box[ii*6+5]);
      Inobox[3] = 1.f / Full_box[ii*6+3];
      Inobox[4] = -Full_box[ii*6+4] / (Full_box[ii*6+3] * Full_box[ii*6+5]);
      Inobox[5] = 1.f / Full_box[ii*6+5];

      for (int i=0;i<N;i++){
        float px=Full_pos[ii*N*3+i*3];
        float py=Full_pos[ii*N*3+i*3+1];
        float pz=Full_pos[ii*N*3+i*3+2];

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
      cudaMemcpy(nowinopos_d+ii*N*3,Nowinopos,N*3*sizeof(float),cudaMemcpyHostToDevice);

      for (int i=0;i<N;i++)
      {
        if (Ime->howmany[i]>Radbuff)
        {
          printf("Buffer radiale saturato by \n");
	  printf("Particle %d at frame %d with %d neighbours \n",i,ii,Ime->howmany[i]);
          exit(1);
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


    float* rad_descr_d=raddescr_tensor->flat<float>().data();
    int* intmap2b_d=intmap2b_tensor->flat<int>().data();
    float* der2b_d=der2b_tensor->flat<float>().data();
    float* des3bsupp_d=des3bsupp_tensor->flat<float>().data();
    float* der3bsupp_d=der3bsupp_tensor->flat<float>().data();
    int* numtriplet_d=numtriplet_tensor->flat<int>().data();

    float* ang_descr_d=angdescr_tensor->flat<float>().data();
    int* intmap3b_d=intmap3b_tensor->flat<int>().data();
    float* der3b_d=der3b_tensor->flat<float>().data();

    fill_radial_launcher(R_c,Radbuff,R_a,Angbuff,N,
                      nowinopos_d,nowbox,
                      howmany_d,with_d,
                      rad_descr_d,intmap2b_d,der2b_d,
                      des3bsupp_d,
                      der3bsupp_d,nf,numtriplet_d,
                      Rs,coeffA,coeffB,coeffC,Pow_alpha,Pow_beta);
    fill_angular_launcher(R_c, Radbuff, R_a, Angbuff, N, nowinopos_d,
		         nowbox, howmany_d, with_d, ang_descr_d,
			 intmap3b_d, des3bsupp_d, der3b_d, der3bsupp_d,
			 nf, numtriplet_d);
     }




   };
   REGISTER_KERNEL_BUILDER(Name("ComputeDescriptorsLight").Device(DEVICE_GPU), ComputeDescriptorsLightOp);
