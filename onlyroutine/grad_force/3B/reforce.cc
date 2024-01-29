#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("ComputeForceTriplGrad")
    .Input("prevgrad: float")
    .Input("netderiv: float")
    .Input("radial_descriptor: float")
    .Input("angular_descriptor: float")
    .Input("descriptor_derivative_rad: float")
    .Input("descriptor_derivative_ang: float")
    .Input("interaction_map_rad: int32")
    .Input("interaction_map_ang: int32")
    .Input("alpha3b_parameters: float")
    .Input("type_emb3b_parameters: float")
    .Input("type_map: int32")
    .Input("tipos: int32")
    .Input("actual_type: int32")
    .Input("num_triplets: int32")
    .Output("gradnet: float")
    .Output("gradalpha: float")
    .Output("gradck: float");

    void gradforce_tripl_Launcher(const float*  prevgrad_T_d,const float*  netderiv_T_d, const float* desr_T_d,
                                      const float* desa_T_d,const float* intderiv_r_T_d,
                                      const float* intderiv_a_T_d,const int* intmap_r_T_d,
                                      const int* intmap_a_T_d,int nr, int na, int N,
                                      int dimbat,int num_finger,const float* type_emb3b_d,int nt,
                                      const int* tipos_T,const int* actual_type,
                                      const int *num_triplets_d,const float* smooth_a_T,
                                      const int* type_map_T_d,int prod,float* gradnet_3b_T_d,
                                      float* grad_alpha3b_T_d,float* grad_emb3b_T_d);


void set_tensor_to_zero_float(float* tensor,int dim);

class ComputeForceTriplGradOp : public OpKernel {
 public:
  explicit ComputeForceTriplGradOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& prevgrad_T = context->input(0);
    const Tensor& netderiv_T = context->input(1);
    const Tensor& desr_T = context->input(2);
    const Tensor& desa_T = context->input(3);
    const Tensor& intderiv_r_T = context->input(4);
    const Tensor& intderiv_a_T = context->input(5);
    const Tensor& intmap_r_T = context->input(6);
    const Tensor& intmap_a_T = context->input(7);
    const Tensor& smooth_a_T = context->input(8);

    const Tensor& type_emb3b_T = context->input(9);
    const Tensor& type_map_T = context->input(10);

    const Tensor& tipos_T = context->input(11);
    const Tensor& actual_type_T = context->input(12);

    const Tensor& num_triplets_T=context->input(13);


    //Grabbing dimension to allocate output
    int dimbat = netderiv_T.shape().dim_size(0);

    int nr = desr_T.shape().dim_size(2);

    int na = desa_T.shape().dim_size(2);

    int N = type_map_T.shape().dim_size(0);
    int Nlocal = desr_T.shape().dim_size(1);

    int nt = tipos_T.shape().dim_size(0);
    int nt_couple=nt*(nt+1)/2;

    const int* actual_type=actual_type_T.flat<int>().data();

    int num_finger=smooth_a_T.shape().dim_size(1);

    //Flatting tensor to be used as arrays
    auto prevgrad_T_d=prevgrad_T.flat<float>();
    auto netderiv_T_d=netderiv_T.flat<float>();
    auto desr_T_d= desr_T.flat<float>();
    auto desa_T_d= desa_T.flat<float>();
    auto intderiv_r_T_d=intderiv_r_T.flat<float>();
    auto intderiv_a_T_d=intderiv_a_T.flat<float>();
    auto intmap_r_T_d=intmap_r_T.flat<int>();
    auto intmap_a_T_d=intmap_a_T.flat<int>();
    auto type_emb3b_T_d=type_emb3b_T.flat<float>();
    auto smooth_a_T_d=smooth_a_T.flat<float>();
    auto type_map_T_d=type_map_T.flat<int>();
    auto tipos_T_d=tipos_T.flat<int>();
    auto num_triplets_T_d=num_triplets_T.flat<int>();

    // Create an output tensor for DL/DT
    Tensor* gradnet_3b_T = NULL;
    TensorShape gradnet_3b_shape ;
    gradnet_3b_shape.AddDim (1);
    gradnet_3b_shape.AddDim (dimbat);
    gradnet_3b_shape.AddDim (Nlocal);
    gradnet_3b_shape.AddDim (num_finger);
    OP_REQUIRES_OK(context, context->allocate_output(0, gradnet_3b_shape,
                                                     &gradnet_3b_T));
    set_tensor_to_zero_float(gradnet_3b_T->flat<float>().data(),dimbat*Nlocal*num_finger);
    // Create an output tensor for DL/Dalpha
    Tensor* grad_alpha3b_T = NULL;
    TensorShape grad_alpha3b_shape ;
    grad_alpha3b_shape.AddDim (nt_couple);
    grad_alpha3b_shape.AddDim (3*num_finger);
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_alpha3b_shape,
                                                     &grad_alpha3b_T));
    int dimnow=nt_couple*3*num_finger;
    set_tensor_to_zero_float(grad_alpha3b_T->flat<float>().data(),dimnow);
    // Create an output tensor for DL/Dck
    Tensor* grad_emb3b_T = NULL;
    TensorShape grad_emb3b_shape;
    grad_emb3b_shape.AddDim (nt_couple);
    grad_emb3b_shape.AddDim (num_finger);
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_emb3b_shape,
                                                     &grad_emb3b_T));
    set_tensor_to_zero_float(grad_emb3b_T->flat<float>().data(),nt_couple*num_finger);

    
    int prod=dimbat*Nlocal*na;
    gradforce_tripl_Launcher(prevgrad_T_d.data(),netderiv_T_d.data(), desr_T_d.data(), desa_T_d.data(),intderiv_r_T_d.data(),intderiv_a_T_d.data(),
                        intmap_r_T_d.data(),intmap_a_T_d.data(),
                        nr, na, N, dimbat,num_finger,type_emb3b_T_d.data(),nt,
                        tipos_T_d.data(),
                        actual_type,num_triplets_T_d.data(),smooth_a_T_d.data(),
                        type_map_T_d.data(),prod,gradnet_3b_T->flat<float>().data(),
                        grad_alpha3b_T->flat<float>().data(),
  			grad_emb3b_T->flat<float>().data());

}
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceTriplGrad").Device(DEVICE_GPU), ComputeForceTriplGradOp);
