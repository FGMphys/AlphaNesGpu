#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include "staf_real.h"

using namespace tensorflow;

void init_block_dim(int buffdim);

REGISTER_OP("InitGradForceRadial")
    .Input("buffdim: int32")
    .Output("code: int32");

    class InitGradForceRadialOp : public OpKernel {
     public:
      explicit InitGradForceRadialOp(OpKernelConstruction* context) : OpKernel(context) {}
      void Compute(OpKernelContext* context) override {
           const Tensor& buffdim = context->input(0);

           init_block_dim(buffdim.flat<int>()(0));

           Tensor* code = NULL;
           TensorShape code_shape ;
           code_shape.AddDim (1);

           OP_REQUIRES_OK(context, context->allocate_output(0, code_shape,
                                                            &code));
            code->flat<int>()(0)=0;


      }
      };
      REGISTER_KERNEL_BUILDER(Name("InitGradForceRadial").Device(DEVICE_CPU), InitGradForceRadialOp);

void set_tensor_to_zero_real(real* tensor_data,int dimension);

REGISTER_OP("ComputeForceRadialGrad")
    .Input("prevgrad: " STAF_TF_DTYPE)
    .Input("netderiv: " STAF_TF_DTYPE)
    .Input("descriptor_derivative_rad: " STAF_TF_DTYPE)
    .Input("interaction_map_rad: int32")
    .Input("radial_descriptor: " STAF_TF_DTYPE)
    .Input("alpha2b_parameters: " STAF_TF_DTYPE)
    .Input("type_emb2b_parameters: " STAF_TF_DTYPE)
    .Input("type_map: int32")
    .Input("tipos: int32")
    .Input("actual_type: int32")
    .Output("gradnet: " STAF_TF_DTYPE)
    .Output("grad_alpha2b: " STAF_TF_DTYPE)
    .Output("grad_emb2b: " STAF_TF_DTYPE);

void back_prop_grad_force2b_Launcher(const real* prevgrad,const real* radiale,
                           int nr,const real* alpha_radiale,int num_finger,
                           const real* desder,const int* intmap2b,
                           int dimbat,int N,int N_local,const real*netderiv,
                           const real* type_emb2b,int nt,const int* type_map,
                           const int* tipos,const int* actual_type,real* grad_net,
                           real* grad_alpha2b,real* grad_emb2b);


class ComputeForceRadialGradOp : public OpKernel {
 public:
  explicit ComputeForceRadialGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& prevgrad_T=context->input(0);
      const Tensor& netderiv_T = context->input(1);
      const Tensor& desder_T = context->input(2);
      const Tensor& intmap2b_T = context->input(3);
      const Tensor& desr_T = context->input(4);
      const Tensor& alpha_radiale_T = context->input(5);

      const Tensor& type_emb2b_T = context->input(6);
      const Tensor& type_map_T = context->input(7);

      const Tensor& tipos_T = context->input(8);
      const Tensor& actual_type_T = context->input(9);


      //Grabbing some useful dimension
      int dimbat = desr_T.shape().dim_size(0);
      int nr = desr_T.shape().dim_size(2);
      int N_local=desr_T.shape().dim_size(1);
      int N = type_map_T.shape().dim_size(0);
      int nt = tipos_T.shape().dim_size(0);
      int num_finger=alpha_radiale_T.shape().dim_size(1);

      //Flatting Tensors
      auto prevgrad=prevgrad_T.flat<real>();
      auto netderiv = netderiv_T.flat<real>();
      auto desder = desder_T.flat<real>();
      auto intmap2b = intmap2b_T.flat<int>();
      auto radiale = desr_T.flat<real>();
      auto alpha_radiale = alpha_radiale_T.flat<real>();
      auto type_emb2b = type_emb2b_T.flat<real>();
      auto type_map = type_map_T.flat<int>();
      auto tipos = tipos_T.flat<int>();

      const int* actual_type=actual_type_T.flat<int>().data();

      // Create an output tensor
      Tensor* grad_net_T = NULL;
      TensorShape grad_net_shape ;

      grad_net_shape.AddDim (dimbat);
      grad_net_shape.AddDim (N_local);
      grad_net_shape.AddDim (num_finger);
      OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                       &grad_net_T));
      set_tensor_to_zero_real(grad_net_T->flat<real>().data(),dimbat*N_local*num_finger);

      Tensor* grad_alpha2b_T = NULL;
      TensorShape grad_alpha2b_shape ;
      grad_alpha2b_shape.AddDim (nt);
      grad_alpha2b_shape.AddDim (num_finger);
      OP_REQUIRES_OK(context, context->allocate_output(1, grad_alpha2b_shape,
                                                       &grad_alpha2b_T));
      set_tensor_to_zero_real(grad_alpha2b_T->flat<real>().data(),nt*num_finger);

      Tensor* grad_emb2b_T = NULL;
      TensorShape grad_emb2b_shape;
      grad_emb2b_shape.AddDim (nt);
      grad_emb2b_shape.AddDim (num_finger);
      OP_REQUIRES_OK(context, context->allocate_output(2,grad_emb2b_shape,
                                                       &grad_emb2b_T));
      set_tensor_to_zero_real(grad_emb2b_T->flat<real>().data(),nt*num_finger);




      back_prop_grad_force2b_Launcher(prevgrad.data(),radiale.data(),nr,alpha_radiale.data(),num_finger,
                           desder.data(),intmap2b.data(),dimbat,N,N_local,netderiv.data(),type_emb2b.data(),
                           nt,type_map.data(),tipos.data(),actual_type,grad_net_T->flat<real>().data(),
                           grad_alpha2b_T->flat<real>().data(),grad_emb2b_T->flat<real>().data());


  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceRadialGrad").Device(DEVICE_GPU), ComputeForceRadialGradOp);
