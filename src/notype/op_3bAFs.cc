#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"


using namespace tensorflow;


void computeSmoothMax3body(const Tensor& ds_T, const Tensor& ds3bod_T, int numdsrad,
                           int numds3bod, Tensor* sds_T,int dimbat, int N,
                           const Tensor& neigh3body_T, const Tensor& smooth_a_T,
                           int nsmooth_a)
{



  auto neigh3body = neigh3body_T.flat<int>();
  auto ds3bod = ds3bod_T.flat<double>();
  auto smooth_a = smooth_a_T.flat<double>();
  auto ds = ds_T.flat<double>();

  auto sds = sds_T->flat<double>();
  for (int k=0;k<dimbat*N*nsmooth_a;k++){
    sds(k)=0.;
  }


  // costruiamo i descrittori
for (int b=0;b<dimbat;b++){
    for (int par=0;par<N;par++){
        int nne3bod=neigh3body(b*N*(numds3bod*2+1)+par*(numds3bod*2+1));
        int actual=b*N*numdsrad+par*numdsrad;
        int actual_ang=b*N*numds3bod+par*numds3bod;
        int aux2=0;
        for (int j=0;j<nne3bod-1;j++){
            for (int y=j+1;y<nne3bod;y++){
                 int neighj=neigh3body(b*N*(numds3bod*2+1)+par*(numds3bod*2+1)+1+aux2*2);
                 int neighy=neigh3body(b*N*(numds3bod*2+1)+par*(numds3bod*2+1)+1+aux2*2+1);



                 double angulardes=ds3bod(actual_ang+aux2);

                 //Funziona solo con due tipi
                 int sum=0;
               for (int a1=0;a1<nsmooth_a;a1++){
                     double betaval=smooth_a(sum*nsmooth_a*3+a1*3+2);
                     double alpha1=smooth_a(sum*nsmooth_a*3+a1*3+0);
                     double alpha2=smooth_a(sum*nsmooth_a*3+a1*3+1);
                     double softmaxweight=exp(alpha1*ds(actual+j)+alpha2*ds(actual+y));
                     softmaxweight+=exp(alpha2*ds(actual+j)+alpha1*ds(actual+y));
                     softmaxweight*=exp(betaval*angulardes);
                     sds(b*nsmooth_a*N+par*nsmooth_a+a1)+=angulardes*softmaxweight/2.;
                    }
               aux2++;
              }
          }
	     }
    }
}






REGISTER_OP("ComputeSortProj3body")
    .Input("angular_descriptor: double")
    .Input("radial_descriptor: double")
    .Input("angular_descriptor_buffsize: int32")
    .Input("radial_descriptor_buffsize: int32")
    .Input("interaction_map_angular: int32")
    .Input("number_of_local_particles: int32")
    .Input("batch_dimension: int32")
    .Input("interaction_map_rad: int32")
    .Input("alpha3b_parameters: double")
    .Input("number_alpha3b: int32")
    .Output("three_body_afs: double");





class ComputeSortProj3bodyOp : public OpKernel {
 public:
  explicit ComputeSortProj3bodyOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& angular_descriptor_T = context->input(0);
    const Tensor& radial_descriptor_T = context->input(1);
    const Tensor& angular_descriptor_buffsize_T = context->input(2);
    const Tensor& radial_descriptor_buffsize_T = context->input(3);
    const Tensor& interaction_map_angular_T = context->input(4);
    const Tensor& number_of_particles_T = context->input(5);
    const Tensor& batch_dimension_T = context->input(6);
    const Tensor& interaction_map_rad_T = context->input(7);
    const Tensor& alpha3b_parameters_T = context->input(8);
    const Tensor& number_alpha3b_T = context->input(9);




    auto dimbat_flat = batch_dimension_T.flat<int>();
    int dimbat=dimbat_flat(0);

    auto numdes2body_flat = radial_descriptor_buffsize_T.flat<int>();
    int numdes2body=numdes2body_flat(0);

    auto numdes3body_flat = angular_descriptor_buffsize_T.flat<int>();
    int numdes3body=numdes3body_flat(0);

    auto N_flat = number_of_particles_T.flat<int>();
    int N=N_flat(0);

    auto nsmooth_a_flat=number_alpha3b_T.flat<int>();
    int nsmooth_a=nsmooth_a_flat(0);


    int dimdes2body=dimbat*numdes2body*N;
    int dimdes3body=dimbat*numdes3body*N;





    // Create an output tensor
    Tensor* three_body_AFs_T = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (dimbat);
    grad_net_shape.AddDim (N);
    grad_net_shape.AddDim (nsmooth_a);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &three_body_AFs_T));

    //Computing three-body atomic-fingerprints
    computeSmoothMax3body(radial_descriptor_T,angular_descriptor_T,numdes2body,numdes3body,
                          three_body_AFs_T,dimbat,N,interaction_map_angular_T,
                          alpha3b_parameters_T,nsmooth_a);


  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeSortProj3body").Device(DEVICE_CPU), ComputeSortProj3bodyOp);
