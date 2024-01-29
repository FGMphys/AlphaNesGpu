#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

REGISTER_OP("ComputeForceRadial")
    .Input("netderiv: float")
    .Input("descriptor_derivative_rad: float")
    .Input("interaction_map_rad: int32")
    .Input("radial_descriptor: float")
    .Input("alpha2b_parameters: float")
    .Input("type_emb2b_parameters: float")
    .Input("type_map: int32")
    .Input("tipos: int32")
    .Input("actual_type: int32")
    .Output("force: float");



void computeforce_doublets_Launcher(const float*  netderiv, const float* des_r,
                    const float* intderiv_r,const int* intmap_r,
                    int nr, int N, int dimbat,int num_alpha_radiale,
                    const float* alpha_radiale,const float* type_emb2b,int nt,
                    const int* tipos_T,const int* actual_type,float* forces2b,const int* type_map,int prod);


void set_tensor_to_zero_float(float* tensor,int dimten);

class ComputeForceRadialOp : public OpKernel {
 public:
  explicit ComputeForceRadialOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& netderiv_T = context->input(0);
    const Tensor& desder_T = context->input(1);
    const Tensor& intmap2b_T = context->input(2);
    const Tensor& desr_T = context->input(3);
    const Tensor& alpha_radiale_T = context->input(4);

    const Tensor& type_emb2b_T = context->input(5);
    const Tensor& type_map_T = context->input(6);

    const Tensor& tipos_T = context->input(7);
    const Tensor& actual_type_T = context->input(8);


    //Grabbing some useful dimension
    int dimbat = netderiv_T.shape().dim_size(0);
    int nr = desr_T.shape().dim_size(2);
    int Nlocal=desr_T.shape().dim_size(1);
    int N = type_map_T.shape().dim_size(0);
    int nt = tipos_T.shape().dim_size(0);
    int num_alpha_radiale=alpha_radiale_T.shape().dim_size(1);

    //Getting data pointer
    auto netderiv_T_flat = netderiv_T.flat<float>();
    auto desder_T_flat = desder_T.flat<float>();
    auto intmap2b_T_flat = intmap2b_T.flat<int>();
    auto desr_T_flat = desr_T.flat<float>();
    auto alpha_radiale_T_flat = alpha_radiale_T.flat<float>();

    auto type_emb2b_T_flat = type_emb2b_T.flat<float>();
    auto type_map_T_flat = type_map_T.flat<int>();

    auto tipos_T_flat = tipos_T.flat<int>();

    const int* actual_type=actual_type_T.flat<int>().data();
    //int actual_type=actual_type_T_flat(0);

    int num_finger=alpha_radiale_T.shape().dim_size(1);

    // Create an output tensor
    Tensor* forces2b_T = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (dimbat);
    grad_net_shape.AddDim (N*3);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &forces2b_T));
   
    set_tensor_to_zero_float(forces2b_T->flat<float>().data(),dimbat*3*N);    
    
    int prod=dimbat*Nlocal*nr;
   computeforce_doublets_Launcher(netderiv_T_flat.data(),desr_T_flat.data(),desder_T_flat.data(),intmap2b_T_flat.data(),nr,N,dimbat,num_alpha_radiale,alpha_radiale_T_flat.data(),type_emb2b_T_flat.data(),nt,tipos_T_flat.data(),actual_type,forces2b_T->flat<float>().data(),type_map_T_flat.data(),prod);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceRadial").Device(DEVICE_GPU), ComputeForceRadialOp);
