#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>



using namespace tensorflow;

void back_prop_grad_forceop(const Tensor& prevgrad_T, const Tensor& ds_T,
                            int numdes,const Tensor& alpha2b_T,
                            int num_alpha_radiale,const Tensor& intderiv_T,
                            const Tensor& intmap2b_T,int dimbat,int N,
                            const Tensor& netderiv_T,Tensor* grad_net_T,
                            Tensor* grad_alpha2b_T)
                            {




auto prevgrad_T_flat=prevgrad_T.flat<double>();
auto ds_T_flat=ds_T.flat<double>();
auto alpha2b_T_flat=alpha2b_T.flat<double>();
auto netderiv_T_flat=netderiv_T.flat<double>();
auto intderiv_T_flat=intderiv_T.flat<double>();
auto intmap2b_T_flat=intmap2b_T.flat<int>();



auto grad_net_T_flat = grad_net_T->flat<double>();
auto grad_alpha2b_T_flat = grad_alpha2b_T->flat<double>();



int N_local=N;

//Set output tensor values to zeros
for (int i = 0; i < (dimbat*N_local*num_alpha_radiale); i++) {
grad_net_T_flat(i)=0.;
}
for (int i = 0; i < (num_alpha_radiale); i++) {
grad_alpha2b_T_flat(i)=0.;
}


for (int b=0; b<dimbat; b++){
     for (int par = 0; par < N_local; par++) {
         int actual=b*N_local*numdes+par*numdes;
         int num_neigh=intmap2b_T_flat(b*(N_local*(numdes+1))+(numdes+1)*par);
         for (int j=0;j<num_neigh;j++){
              int neighj=intmap2b_T_flat(b*(N_local*(numdes+1))+(numdes+1)*par+1+j);
              double ds_el=ds_T_flat(actual+j);
              for (int a =0; a<3; a++){
                double prevgrad_el=prevgrad_T_flat(b*(N*3)+par*3+a);
                double prevgrad_neigh=prevgrad_T_flat(b*(N*3)+neighj*3+a);
                double common = 0.5*intderiv_T_flat(b*N_local*3*numdes+numdes*3*par+a*numdes+j);

                 for (int i=0;i<num_alpha_radiale;i++){
                      double alpha_el=alpha2b_T_flat(i);
                      double supp1=exp(alpha_el*ds_el);
                      double sds_deriv=supp1*(1.+alpha_el*ds_el);
                      double buff_alpha=supp1*ds_el*(2.+alpha_el*ds_el);

                      double  NGel=netderiv_T_flat(b*N_local*num_alpha_radiale+par*num_alpha_radiale+i);
                      int index_sup=b*(N_local*num_alpha_radiale)+par*num_alpha_radiale+i;

                      grad_net_T_flat(index_sup) -= prevgrad_el*common*sds_deriv;
                      grad_net_T_flat(index_sup) += prevgrad_neigh*common*sds_deriv;

                      grad_alpha2b_T_flat(i)-=prevgrad_el*NGel*buff_alpha*common;
                      grad_alpha2b_T_flat(i)+=prevgrad_neigh*NGel*buff_alpha*common;


               }
        }




               }
           }
         }
}
























REGISTER_OP("ComputeForceRadialGrad")
    .Input("prevgrad: double")
    .Input("netderiv: double")
    .Input("descriptor_derivative_rad: double")
    .Input("interaction_map_rad: int32")
    .Input("number_of_particles_in_frame: int32")
    .Input("radial_descriptor_buffsize: int32")
    .Input("batch_dimension: int32")
    .Input("radial_descriptor: double")
    .Input("number_alpha2b: int32")
    .Input("alpha2b_parameters: double")
    .Output("gradnet: double")
    .Output("grad_alpha2b: double");





class ComputeForceRadialGradOp : public OpKernel {
 public:
  explicit ComputeForceRadialGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
       // Grab the input tensor
       const Tensor& prevgrad_T=context->input(0);
       const Tensor& netderiv_T = context->input(1);
       const Tensor& desder_T = context->input(2);
       const Tensor& intmap2b_T = context->input(3);
       const Tensor& numpar_T = context->input(4);
       const Tensor& des_buffsize_T = context->input(5);
       const Tensor& dimbatch_T = context->input(6);
       const Tensor& radiale_T = context->input(7);
       const Tensor& num_alpha2b_T = context->input(8);
       const Tensor& alpha_radiale_T = context->input(9);



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



    // Create an output tensor
    Tensor* grad_net_T = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (1);
    grad_net_shape.AddDim (dimbat);
    grad_net_shape.AddDim (N);
    grad_net_shape.AddDim (num_alpha_radiale);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &grad_net_T));

    Tensor* grad_alpha2b_T = NULL;
    TensorShape grad_alpha2b_shape ;
    grad_alpha2b_shape.AddDim (num_alpha_radiale);
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_alpha2b_shape,
                                                     &grad_alpha2b_T));




    back_prop_grad_forceop(prevgrad_T,radiale_T,numdes,alpha_radiale_T,num_alpha_radiale,
                           desder_T,intmap2b_T,dimbat,N,netderiv_T,grad_net_T,grad_alpha2b_T);


  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceRadialGrad").Device(DEVICE_CPU), ComputeForceRadialGradOp);
