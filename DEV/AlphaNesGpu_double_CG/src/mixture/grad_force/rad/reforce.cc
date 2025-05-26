#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>

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

void set_tensor_to_zero_double(double* tensor_data,int dimension);

REGISTER_OP("ComputeForceRadialGrad")
    .Input("prevgrad: double")
    .Input("netderiv: double")
    .Input("descriptor_derivative_rad: double")
    .Input("interaction_map_rad: int32")
    .Input("radial_descriptor: double")
    .Input("alpha2b_parameters: double")
    .Input("type_emb2b_parameters: double")
    .Input("color_type_map: int32")
    .Input("map_color_interaction: int32")
    .Input("actual_type: int32")
    .Input("map_intra: int32")
    .Output("gradnet: double")
    .Output("grad_alpha2b: double")
    .Output("grad_emb2b: double");

void back_prop_grad_force2b_Launcher(const double* prevgrad,const double* radiale,
                           int nr,const double* alpha_radiale,int num_finger,
                           const double* desder,const int* intmap2b,
                           int dimbat,int N,int N_local,const double*netderiv,
                           const double* type_emb2b,int nt,const int* color_type_map,
                           const int* map_color_interaction,const int* actual_type,
                           const int* map_intra,double* grad_net,double* grad_alpha2b,
                           double* grad_emb2b);


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
      const Tensor& color_type_map_T = context->input(7);

      const Tensor& map_color_interaction_T = context->input(8);
      const Tensor& actual_type_T = context->input(9);

      const Tensor& map_intra_T = context->input(10);


      //Grabbing some useful dimension
      int dimbat = desr_T.shape().dim_size(0);
      int nr = desr_T.shape().dim_size(2);
      int N_local=desr_T.shape().dim_size(1);
      int N = color_type_map_T.shape().dim_size(0);
      int nt = alpha_radiale_T.shape().dim_size(0);
      int num_finger=alpha_radiale_T.shape().dim_size(1);

      //Flatting Tensors
      auto prevgrad=prevgrad_T.flat<double>();
      auto netderiv = netderiv_T.flat<double>();
      auto desder = desder_T.flat<double>();
      auto intmap2b = intmap2b_T.flat<int>();
      auto radiale = desr_T.flat<double>();
      auto alpha_radiale = alpha_radiale_T.flat<double>();
      auto type_emb2b = type_emb2b_T.flat<double>();
      auto color_type_map = type_map_T.flat<int>();
      auto map_color_interaction=map_color_interaction_T.flat<int>();
      auto map_intra=map_intra_T.flat<int>();

      const int* actual_type=actual_type_T.flat<int>().data();

      // Create an output tensor
      Tensor* grad_net_T = NULL;
      TensorShape grad_net_shape ;

      grad_net_shape.AddDim (dimbat);
      grad_net_shape.AddDim (N_local);
      grad_net_shape.AddDim (num_finger);
      OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                       &grad_net_T));
      set_tensor_to_zero_double(grad_net_T->flat<double>().data(),dimbat*N_local*num_finger);

      Tensor* grad_alpha2b_T = NULL;
      TensorShape grad_alpha2b_shape ;
      grad_alpha2b_shape.AddDim (nt);
      grad_alpha2b_shape.AddDim (num_finger);
      OP_REQUIRES_OK(context, context->allocate_output(1, grad_alpha2b_shape,
                                                       &grad_alpha2b_T));
      set_tensor_to_zero_double(grad_alpha2b_T->flat<double>().data(),nt*num_finger);

      Tensor* grad_emb2b_T = NULL;
      TensorShape grad_emb2b_shape;
      grad_emb2b_shape.AddDim (nt);
      grad_emb2b_shape.AddDim (num_finger);
      OP_REQUIRES_OK(context, context->allocate_output(2,grad_emb2b_shape,
                                                       &grad_emb2b_T));
      set_tensor_to_zero_double(grad_emb2b_T->flat<double>().data(),nt*num_finger);




      back_prop_grad_force2b_Launcher(prevgrad.data(),radiale.data(),nr,alpha_radiale.data(),num_finger,
                           desder.data(),intmap2b.data(),dimbat,N,N_local,netderiv.data(),type_emb2b.data(),
                           nt,color_type_map.data(),map_color_interaction.data(),actual_type,map_intra.data(),
                           grad_net_T->flat<double>().data(),grad_alpha2b_T->flat<double>().data(),
                           grad_emb2b_T->flat<double>().data());


  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceRadialGrad").Device(DEVICE_GPU), ComputeForceRadialGradOp);
