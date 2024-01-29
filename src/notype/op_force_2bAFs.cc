#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"


using namespace tensorflow;

void computeSmoothMaxForce(const Tensor& ds_T,int num_ds,const Tensor& alpha2b_T,
                          int num_alpha_radiale,const Tensor& netderiv_T,
                          const Tensor& intderiv_T, const Tensor& intmap2b_T,
                          int dimbat,int N,Tensor* forces2b)
{
  int numdes=num_ds;

  auto ds_T_flat=ds_T.flat<double>();
  auto alpha2b_T_flat=alpha2b_T.flat<double>();
  auto netderiv_T_flat=netderiv_T.flat<double>();
  auto intderiv_T_flat=intderiv_T.flat<double>();
  auto intmap2b_T_flat=intmap2b_T.flat<int>();




  auto forces2bflat = forces2b->flat<double>();
  //Set output tensor values to zeros
  for (int i = 0; i < (dimbat*N*3); i++) {
  forces2bflat(i)=0.;
  }

  int N_local=N;

for (int b=0;b<dimbat;b++){

    for (int par2=0;par2<N_local;par2++){
        int actual=b*N_local*numdes+par2*numdes;
        int num_neigh=intmap2b_T_flat(b*(N_local*(numdes+1))+(numdes+1)*par2);
        for (int j=0;j<num_neigh;j++)
        {

          double des_r=ds_T_flat(actual+j);
          int neighj=intmap2b_T_flat(b*(N_local*(numdes+1))+(numdes+1)*par2+1+j);

          for (int a =0; a<3; a++){
              double intder_r=intderiv_T_flat(b*N_local*3*numdes+numdes*3*par2+a*numdes+j);
              for (int i=0; i<num_alpha_radiale;i++){
                  double alpha_now=alpha2b_T_flat(i);
                  double sds_deriv=exp(alpha_now*des_r);
                  sds_deriv*=(1.+alpha_now*des_r);
                  double prevgrad=netderiv_T_flat(b*N_local*num_alpha_radiale+num_alpha_radiale*par2+i);
                  double temp = 0.5*sds_deriv*intder_r;
                  forces2bflat(b*(N*3)+3*par2+a) -= prevgrad*temp;
                  forces2bflat(b*(N*3)+3*neighj+a) += prevgrad*temp;
                 }
               }

            }
}
}
}


REGISTER_OP("ComputeForceRadial")
    .Input("netderiv: double")
    .Input("descriptor_derivative_rad: double")
    .Input("interaction_map_rad: int32")
    .Input("number_of_particles_in_frame: int32")
    .Input("radial_descriptor_buffsize: int32")
    .Input("batch_dimension: int32")
    .Input("radial_descriptor: double")
    .Input("number_alpha2b: int32")
    .Input("alpha2b_parameters: double")
    .Output("force: double");


class ComputeForceRadialOp : public OpKernel {
 public:
  explicit ComputeForceRadialOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& netderiv_T = context->input(0);
    const Tensor& desder_T = context->input(1);
    const Tensor& intmap2b_T = context->input(2);
    const Tensor& numpar_T = context->input(3);
    const Tensor& des_buffsize_T = context->input(4);
    const Tensor& dimbatch_T = context->input(5);
    const Tensor& radiale_T = context->input(6);
    const Tensor& num_alpha2b_T = context->input(7);
    const Tensor& alpha_radiale_T = context->input(8);



    //Copio i tensori input in nuovi array per elaborarli
    auto dimbatch_T_flat = dimbatch_T.flat<int>();
    auto des_buffsize_T_flat = des_buffsize_T.flat<int>();
    auto numpar_T_flat = numpar_T.flat<int>();
    auto num_alpha2b_T_flat = num_alpha2b_T.flat<int>();

    int dimbat=dimbatch_T_flat(0);
    int numdes=des_buffsize_T_flat(0);
    int N=numpar_T_flat(0);
    int dimdes=dimbat*numdes*N;
    int num_alpha_radiale=num_alpha2b_T_flat(0);
    int dimnetderiv=num_alpha_radiale*dimbat*N;







    // Create an output tensor
    Tensor* forces2b_T = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (dimbat);
    grad_net_shape.AddDim (N);
    grad_net_shape.AddDim (3);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &forces2b_T));

    computeSmoothMaxForce(radiale_T,numdes,alpha_radiale_T,num_alpha_radiale,
                          netderiv_T,desder_T,intmap2b_T,dimbat,N,forces2b_T);



  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceRadial").Device(DEVICE_CPU), ComputeForceRadialOp);
