#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"



using namespace tensorflow;
void compute_nextgrad(const Tensor& descriptor2b_T, const Tensor& descriptor3b_T,
                      int numdsrad, int numds3bod,const Tensor& prevgrad_T,
                      int dimbat, int N, const Tensor& intmap3b_T,
                      const Tensor& alpha3b_T, int nsmooth_a,Tensor* next_alpha3b_grad_T)
{

auto neigh3body=intmap3b_T.flat<int>();
auto ds=descriptor2b_T.flat<double>();
auto ds3bod=descriptor3b_T.flat<double>();
auto smooth_a=alpha3b_T.flat<double>();
auto prevgrad=prevgrad_T.flat<double>();

auto nextgrad=next_alpha3b_grad_T->flat<double>();

for (int k=0;k<nsmooth_a*3;k++){
     nextgrad(k)=0.;
}


for (int b=0;b<dimbat;b++){

    for (int par=0;par<N;par++){
        int nne3bod=neigh3body(b*N*(numds3bod*2+1)+par*(numds3bod*2+1));
        int actual=b*N*numdsrad+par*numdsrad;
        int actual_ang=b*N*numds3bod+par*numds3bod;
        int aux2=0;
        //Costruzione della proiezione sia della parte 2body  che 3body....////
       ////////////////////////////////////////////////////////////////////////
        for (int j=0;j<nne3bod-1;j++){
           for (int y=j+1;y<nne3bod;y++){
               int neighj=neigh3body(b*N*(numds3bod*2+1)+par*(numds3bod*2+1)+1+aux2*2);
               int neighy=neigh3body(b*N*(numds3bod*2+1)+par*(numds3bod*2+1)+1+aux2*2+1);

               int sum=0;

               double dj=ds(actual+j);
               double dy=ds(actual+y);
               double Tjy=ds3bod(actual_ang+aux2);

               for (int a1=0;a1<nsmooth_a;a1++){
                    double betaval=smooth_a(sum*nsmooth_a*3+a1*3+2);
                    double alpha1=smooth_a(sum*nsmooth_a*3+a1*3+0);
                    double alpha2=smooth_a(sum*nsmooth_a*3+a1*3+1);
                    //NUMERATORE//
                    double prevgradel=prevgrad(b*nsmooth_a*N+par*nsmooth_a+a1);
                     //Derivate rispetto beta gamma delta
                     double a1dja2dy=exp(alpha1*dj+alpha2*dy);
                     double a1dya2dj=exp(alpha1*dy+alpha2*dj);
                     double btjy=exp(betaval*Tjy);


                     nextgrad(sum*nsmooth_a*3+a1*3+0)+=prevgradel*(a1dja2dy*dj+a1dya2dj*dy)*btjy*Tjy/2.;
                     nextgrad(sum*nsmooth_a*3+a1*3+1)+=prevgradel*(a1dja2dy*dy+a1dya2dj*dj)*btjy*Tjy/2.;
                     nextgrad(sum*nsmooth_a*3+a1*3+2)+=prevgradel*(a1dja2dy+a1dya2dj)*btjy*Tjy*Tjy/2.;


                   }
                   aux2++;
              }

	    }
	}
    }

  }






using namespace tensorflow;

REGISTER_OP("ComputeSortProj3bodyGrad")
    .Input("previous_gradient: double")
    .Input("angular_desciptors: double")
    .Input("radial_descriptors: double")
    .Input("numdes3body: int32")
    .Input("numdes2body: int32")
    .Input("intmap3b: int32")
    .Input("number_of_particles: int32")
    .Input("dimbatch: int32")
    .Input("intmap2b: int32")
    .Input("alpha3b_parameters: double")
    .Input("number_of_alpha3b: int32")
    .Output("alphagrad3body: double");





class ComputeSortProj3bodyGradOp : public OpKernel {
 public:
  explicit ComputeSortProj3bodyGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& prevgrad_T = context->input(0);
    const Tensor& descriptor3b_T = context->input(1);
    const Tensor& descriptor2b_T = context->input(2);
    const Tensor& numdes3body_T = context->input(3);
    const Tensor& numdes2body_T = context->input(4);
    const Tensor& intmap3b_T = context->input(5);
    const Tensor& numpar_T = context->input(6);
    const Tensor& dimbatch_T = context->input(7);
    const Tensor& intmap2b_T = context->input(8);
    const Tensor& alpha3b_T = context->input(9);
    const Tensor& nalpha3b_T = context->input(10);


    //Dimension of tensor
    auto dimbatch_flat=dimbatch_T.flat<int>();
    auto numdes2body_flat=numdes2body_T.flat<int>();
    auto numdes3body_flat=numdes3body_T.flat<int>();
    auto numpar_flat=numpar_T.flat<int>();
    auto nalpha3b_flat=nalpha3b_T.flat<int>();





    int dimbat=dimbatch_flat(0);
    int numdes2body=numdes2body_flat(0);
    int numdes3body=numdes3body_flat(0);
    int N=numpar_flat(0);
    int dimdes2body=dimbat*numdes2body*N;
    int dimdes3body=dimbat*numdes3body*N;
    int nsmooth_a=nalpha3b_flat(0);




    // Create an output tensor for gradient wrt beta gamma delta AFs
    Tensor* next_alpha3b_grad_T = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (nsmooth_a);
    grad_net_shape.AddDim (3);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &next_alpha3b_grad_T));


    compute_nextgrad(descriptor2b_T,descriptor3b_T,numdes2body,numdes3body,prevgrad_T,dimbat,
                     N,intmap3b_T,alpha3b_T,nsmooth_a,next_alpha3b_grad_T);



  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeSortProj3bodyGrad").Device(DEVICE_CPU), ComputeSortProj3bodyGradOp);
