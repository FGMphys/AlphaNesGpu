#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>

using namespace tensorflow;

void set_tensor_to_zero_float(float* tensor_data,int dimension);

REGISTER_OP("ComputeForceRadialGrad")
    .Input("prevgrad: float")
    .Input("netderiv: float")
    .Input("descriptor_derivative_rad: float")
    .Input("interaction_map_rad: int32")
    .Input("radial_descriptor: float")
    .Input("alpha2b_parameters: float")
    .Input("type_emb2b_parameters: float")
    .Input("type_map: int32")
    .Input("tipos: int32")
    .Input("actual_type: int32")
    .Output("gradnet: float")
    .Output("grad_alpha2b: float")
    .Output("grad_emb2b: float");

void back_prop_grad_force2b_Launcher(const float* prevgrad,const float* radiale,
                           int nr,const float* alpha_radiale,int num_finger,
                           const float* desder,const int* intmap2b,
                           int dimbat,int N,int N_local,const float*netderiv,
                           const float* type_emb2b,int nt,const int* type_map,
                           const int* tipos,const int* actual_type,float* grad_net,
                           float* grad_alpha2b,float* grad_emb2b);


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
      auto prevgrad=prevgrad_T.flat<float>();
      auto netderiv = netderiv_T.flat<float>();
      auto desder = desder_T.flat<float>();
      auto intmap2b = intmap2b_T.flat<int>();
      auto radiale = desr_T.flat<float>();
      auto alpha_radiale = alpha_radiale_T.flat<float>();
      auto type_emb2b = type_emb2b_T.flat<float>();
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
      set_tensor_to_zero_float(grad_net_T->flat<float>().data(),dimbat*N_local*num_finger);

      Tensor* grad_alpha2b_T = NULL;
      TensorShape grad_alpha2b_shape ;
      grad_alpha2b_shape.AddDim (nt);
      grad_alpha2b_shape.AddDim (num_finger);
      OP_REQUIRES_OK(context, context->allocate_output(1, grad_alpha2b_shape,
                                                       &grad_alpha2b_T));
      set_tensor_to_zero_float(grad_alpha2b_T->flat<float>().data(),nt*num_finger);

      Tensor* grad_emb2b_T = NULL;
      TensorShape grad_emb2b_shape;
      grad_emb2b_shape.AddDim (nt);
      grad_emb2b_shape.AddDim (num_finger);
      OP_REQUIRES_OK(context, context->allocate_output(2,grad_emb2b_shape,
                                                       &grad_emb2b_T));
      set_tensor_to_zero_float(grad_emb2b_T->flat<float>().data(),nt*num_finger);




      back_prop_grad_force2b_Launcher(prevgrad.data(),radiale.data(),nr,alpha_radiale.data(),num_finger,
                           desder.data(),intmap2b.data(),dimbat,N,N_local,netderiv.data(),type_emb2b.data(),
                           nt,type_map.data(),tipos.data(),actual_type,grad_net_T->flat<float>().data(),
                           grad_alpha2b_T->flat<float>().data(),grad_emb2b_T->flat<float>().data());


  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceRadialGrad").Device(DEVICE_GPU), ComputeForceRadialGradOp);
