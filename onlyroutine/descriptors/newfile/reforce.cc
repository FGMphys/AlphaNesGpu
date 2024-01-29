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
#include "log.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#define PI 3.141592654f

static int Radbuff,Angbuff;
static float R_c,Rs,R_a,coeffA,coeffB,coeffC,Pow_alpha,Pow_beta;

static float box[6],inobox[6];
static interactionmap *ime;


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

void construct_descriptor(float* box,int N){

          inobox[0]=1./box[0];
          inobox[1]=-box[1]/(box[0]*box[3]);
          inobox[2]=(box[1]*box[4])/(box[0]*box[3]*box[5])-box[2]/(box[0]*box[5]);
          inobox[3]=1./box[3];
          inobox[4]=-box[4]/(box[3]*box[5]);
          inobox[5]=1./box[5];

          cells=getList(box,R_c,N);

          // INTERACTION MAPS
          ime=createInteractionMap(N,Radbuff);

 }

using namespace tensorflow;

REGISTER_OP("ConstructDescriptorsLight")
    .Input("radial_cutoff: float")
    .Input("radial_buffer: int32")
    .Input("angular_buffer: int32")
    .Input("numpar: int32")
    .Input("boxer: float")
    .Input("rs: float")
    .Input("ra: float")
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


           auto rs_T_flat=rs_T.flat<float>();
           Rs=rs_T_flat(0);

           auto rcrad_T_flat=rcrad_T.flat<float>();
           R_c=rcrad_T_flat(0);

           auto radbuff_T_flat=radbuff_T.flat<int>();
           Radbuff=radbuff_T_flat(0);

           auto angbuff_T_flat=angbuff_T.flat<int>();
           Angbuff=angbuff_T_flat(0);

           auto ra_T_flat=rarad_T.flat<float>();
           R_a=rarad_T_flat(0);

           construct_repulsion();
           construct_descriptor(box_T.flat<float>().data(),numpar_T.flat<int>()(0));
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
    auto boxes = box_T.flat<float>();



    //Copio i tensori input in nuovi array per elaborarli
    int nf=boxes_T.shape().dim_size(0);
    int N=int(positions_T.shape().dim_size(1)/3);

    //////////BUILDING CELL LIST AND IME (FULL ORDERED INTERACTION MAP)////
    // loop per il calcolo di inobox, posizioni in spazio reale, e mappe di vicini per tutti i frame
    int ii;
    for (ii=0;ii<nf;ii++)
    {
      nowinobox[ii*6+0]=1./nowbox[ii*6+0];
      nowinobox[ii*6+1]=-nowbox[ii*6+1]/(nowbox[ii*6+0]*nowbox[ii*6+3]);
      nowinobox[ii*6+2]=(nowbox[ii*6+1]*nowbox[ii*6+4])/(nowbox[ii*6+0]*nowbox[ii*6+3]*nowbox[ii*6+5])-nowbox[ii*6+2]/(nowbox[ii*6+0]*nowbox[ii*6+5]);
      nowinobox[ii*6+3]=1./nowbox[ii*6+3];
      nowinobox[ii*6+4]=-nowbox[ii*6+4]/(nowbox[ii*6+3]*nowbox[ii*6+5]);
      nowinobox[ii*6+5]=1./nowbox[ii*6+5];

      for (int i=0;i<N;i++){
        float px=nowpos[ii*N*3+i*3];
        float py=nowpos[ii*N*3+i*3+1];
        float pz=nowpos[ii*N*3+i*3+2];

        nowpos[ii*N*3+i*3]=(inobox[ii*6+0]*px+inobox[ii*6+1]*py+inobox[ii*6+2]*pz);
        nowpos[ii*N*3+i*3+1]=(inobox[ii*6+3]*py+inobox[ii*6+4]*pz);
        nowpos[ii*N*3+i*3+2]=(inobox[ii*6+5]*pz);
      }

      // calcolo delle celle e dei neighbour list
      fullUpdateList(cells,pos,N,&nowbox[ii*6],R_c);
      resetInteractionMap(ime);
      calculateInteractionMapWithCutoffDistanceOrdered(cells,ime,pos,&nowbox[ii*6],R_c);

      cudaMemcpy(howmany_d+ii*N,ime->howmany,N*sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(with_d+ii*N*radbuff,ime->with[0],N*radbuff*sizeof(int),cudaMemcpyHostToDevice);
      for (int i=0;i<N;i++)
      {
        if (ime->howmany[i]>Radbuff)
        {
          printf("Buffer radiale saturato by \n");
	  printf("Particle %d at frame %d with %d neighbours \n",i,ii,ime->howmany[i]);
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
    raddescr_shape.AddDim (radial_buffer);
    OP_REQUIRES_OK(context, context->allocate_output(0,raddescr_shape,
                                                     &raddescr_tensor));

    // Create an output tensor
    Tensor* angdescr_tensor = NULL;
    TensorShape angdescr_shape;
    angdescr_shape.AddDim (nf);
    angdescr_shape.AddDim (N);
    angdescr_shape.AddDim (angular_buffer);
    OP_REQUIRES_OK(context, context->allocate_output(1,angdescr_shape,
                                                     &angdescr_tensor));


    ///////////////DESCRIPTORS 3B SUPP///////////////
    // Create an output tensor
    Tensor* des3bsupp_tensor = NULL;
    TensorShape des3bsupp_shape;
    des3bsupp_shape.AddDim (nf);
    des3bsupp_shape.AddDim (N);
    des3bsupp_shape.AddDim (radial_buffer);
    OP_REQUIRES_OK(context, context->allocate_output(2,des3bsupp_shape,
                                                     &des3bsupp_tensor));
    ///////////////INTMAP2B///////////////
    // Create an output tensor
    Tensor* intmap2b_tensor = NULL;
    TensorShape intmap2b_shape;
    intmap2b_shape.AddDim (nf);
    intmap2b_shape.AddDim (N);
    intmap2b_shape.AddDim (radial_buffer+1);
    OP_REQUIRES_OK(context, context->allocate_output(3,intmap2b_shape,
                                                     &intmap2b_tensor));
    /////////////////////////////
    ///////////////INTMAP3B///////////////
    // Create an output tensor
    Tensor* intmap3b_tensor = NULL;
    TensorShape intmap3b_shape;
    intmap3b_shape.AddDim (nf);
    intmap3b_shape.AddDim (N);
    intmap3b_shape.AddDim (ang_buffer*2);
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
    der2b_shape.AddDim (radial_buffer);
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
    der3b_shape.AddDim (ang_buffer*2);
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
    der3bsupp_shape.AddDim (radial_buffer);
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
    float* der2b_d=der2b_tensot->flat<float>().data();
    float* des3bsupp_d=des3bsupp_tensor->flat<float>().data();
    float* der3bsupp_d=der3bsupp_tensor->flat<float>().data();
    int* numtriplet_d=numtriplet_tensor->flat<int>().data();

    float* ang_descr_d=angdescr_tensor->flat<float>().data();
    int* intmap3b_d=intmap3b_tensor->flat<int>().data();
    float* der3b_d=der3b_tensor->flat<float>().data();
    int* numtriplet_d=numtriplet_tensor->flat<int>().data();


    fill_radial_launcher(R_c,Radbuff,R_a,Angbuff,N,
                      inopos_d,box_d,
                      howmany_d,with_d,
                      rad_descr_d,intmap2b_d,der2b_d,
                      des3bsupp_d,
                      der3bsupp_d,nf,numtriplet_d,
                      Rs,coeffA,coeffB,coeffC,Pow_alpha,Pow_beta);
    fill_angular_launcher(R_c, Radbuff, R_a, Angbuff, N, inopos_d,
		         box_d, howmany_d, with_d, ang_descr_d,
			 intmap3b_d, des3bsupp_d, der3b_d, der3bsupp_d,
			 nf, numtriplet_d);
     }




   };
   REGISTER_KERNEL_BUILDER(Name("ComputeDescriptorsLight").Device(DEVICE_GPU), ComputeDescriptorsLightOp);
