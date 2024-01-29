///Implementazione del gradiente di una funzione scalare L(SD), funzione dei SD(alpha).
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"


using namespace tensorflow;

REGISTER_OP("ComputeTwoBodyParGrad")
    .Input("prev_grad: float")
    .Input("radial_descriptor: float")
    .Input("interaction_map_rad: int32")
    .Input("alpha2b_parameters: float")
    .Input("type_emb2b_parameters: float")
    .Input("type_map: int32")
    .Output("nextgrad_alpha2b: float")
    .Output("nextgrad_emb2b: float");

void alpha_dist_grad_Launcher(const float* radial_descriptor,int nr,
                       const float* alpha2b_parameters,
                       int nalpha_r,float* nextgrad_alpha2b,int dimbat,
                       int Nlocal,const int* interaction_map_rad,
                       const float* prev_grad,const float* type_emb2b,
                       const int* type_map,float* nextgrad_emb2);
void set_tensor_to_zero_float(float* tensor,int dimten);

class ComputeTwoBodyParGradOp : public OpKernel {
 public:
  explicit ComputeTwoBodyParGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& prev_grad_T = context->input(0);
    const Tensor& radiale_T = context->input(1);
    const Tensor& intmap2b_T = context->input(2);
    const Tensor& alpha_radiale_T = context->input(3);
    const Tensor& type_emb2b_T = context->input(4);
    const Tensor& type_map_T = context->input(5);

    //flattizzo
    auto prev_grad=prev_grad_T.flat<float>();
    auto radial_descriptor = radiale_T.flat<float>();
    auto interaction_map_rad = intmap2b_T.flat<int>();
    auto alpha2b_parameters = alpha_radiale_T.flat<float>();

    auto type_emb2b = type_emb2b_T.flat<float>();
    auto type_map = type_map_T.flat<int>();


    //Prendo le dimensioni del tensore
    int dimbat = radiale_T.shape().dim_size(0);
    int nr = radiale_T.shape().dim_size(2);
    int Nlocal = radiale_T.shape().dim_size(1);
    int nalpha_r=alpha_radiale_T.shape().dim_size(1);
    int nt = alpha_radiale_T.shape().dim_size(0);
    int dimdes=dimbat*nr*Nlocal;


    // Create an output tensor for BackProp of alphas
    Tensor* nextgrad_alpha2b_T = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (nt);
    grad_net_shape.AddDim (nalpha_r);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &nextgrad_alpha2b_T));
    set_tensor_to_zero_float(nextgrad_alpha2b_T->flat<float>().data(),nt*nalpha_r);

    //Create output tensor for backprob of embedding 2b params
    Tensor* nextgrad_emb2_T = NULL;
    TensorShape grad_net_shape2 ;
    grad_net_shape2.AddDim (nt);
    grad_net_shape2.AddDim (nalpha_r);
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_net_shape2,
                                                     &nextgrad_emb2_T));
    set_tensor_to_zero_float(nextgrad_emb2_T->flat<float>().data(),nt*nalpha_r);

    alpha_dist_grad_Launcher(radial_descriptor.data(),nr,alpha2b_parameters.data(),
                           nalpha_r,nextgrad_alpha2b_T->flat<float>().data(),dimbat,
                           Nlocal,interaction_map_rad.data(),
                           prev_grad.data(),type_emb2b.data(),type_map.data(),
                           nextgrad_emb2_T->flat<float>().data());

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeTwoBodyParGrad").Device(DEVICE_GPU), ComputeTwoBodyParGradOp);
