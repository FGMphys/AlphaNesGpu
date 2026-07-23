///Implementazione del gradiente di una funzione scalare L(SD), funzione dei SD(alpha).
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "staf_real.h"

#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"



using namespace tensorflow;

REGISTER_OP("ComputeTwoBodyParGrad")
    .Input("prev_grad: " STAF_TF_DTYPE)
    .Input("radial_descriptor: " STAF_TF_DTYPE)
    .Input("interaction_map_rad: int32")
    .Input("alpha2b_parameters: " STAF_TF_DTYPE)
    .Input("type_emb2b_parameters: " STAF_TF_DTYPE)
    .Input("type_map: int32")
    .Output("nextgrad_alpha2b: " STAF_TF_DTYPE)
    .Output("nextgrad_emb2b: " STAF_TF_DTYPE);

void alpha_dist_grad_Launcher(const real* radial_descriptor,int nr,
                       const real* alpha2b_parameters,
                       int nalpha_r,real* nextgrad_alpha2b,int dimbat,
                       int Nlocal,const int* interaction_map_rad,
                       const real* prev_grad,const real* type_emb2b,
                       const int* type_map,real* nextgrad_emb2, cudaStream_t stream);
void set_tensor_to_zero_real(real* tensor,int dimten, cudaStream_t stream);

class ComputeTwoBodyParGradOp : public OpKernel {
 public:
  explicit ComputeTwoBodyParGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const cudaStream_t stream = context->eigen_device<Eigen::GpuDevice>().stream();

    // Grab the input tensor
    const Tensor& prev_grad_T = context->input(0);
    const Tensor& radiale_T = context->input(1);
    const Tensor& intmap2b_T = context->input(2);
    const Tensor& alpha_radiale_T = context->input(3);
    const Tensor& type_emb2b_T = context->input(4);
    const Tensor& type_map_T = context->input(5);

    //flattizzo
    auto prev_grad=prev_grad_T.flat<real>();
    auto radial_descriptor = radiale_T.flat<real>();
    auto interaction_map_rad = intmap2b_T.flat<int>();
    auto alpha2b_parameters = alpha_radiale_T.flat<real>();

    auto type_emb2b = type_emb2b_T.flat<real>();
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
    set_tensor_to_zero_real(nextgrad_alpha2b_T->flat<real>().data(),nt*nalpha_r, stream);

    //Create output tensor for backprob of embedding 2b params
    Tensor* nextgrad_emb2_T = NULL;
    TensorShape grad_net_shape2 ;
    grad_net_shape2.AddDim (nt);
    grad_net_shape2.AddDim (nalpha_r);
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_net_shape2,
                                                     &nextgrad_emb2_T));
    set_tensor_to_zero_real(nextgrad_emb2_T->flat<real>().data(),nt*nalpha_r, stream);

    alpha_dist_grad_Launcher(radial_descriptor.data(),nr,alpha2b_parameters.data(),
                           nalpha_r,nextgrad_alpha2b_T->flat<real>().data(),dimbat,
                           Nlocal,interaction_map_rad.data(),
                           prev_grad.data(),type_emb2b.data(),type_map.data(),
                           nextgrad_emb2_T->flat<real>().data(), stream);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeTwoBodyParGrad").Device(DEVICE_GPU), ComputeTwoBodyParGradOp);
