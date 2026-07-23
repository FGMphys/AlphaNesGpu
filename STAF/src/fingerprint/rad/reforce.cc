#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "staf_real.h"

#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"



using namespace tensorflow;

REGISTER_OP("ComputeSortProj")
    .Input("radial_descriptor: " STAF_TF_DTYPE)
    .Input("interaction_map_rad: int32")
    .Input("alpha2b_parameters: " STAF_TF_DTYPE)
    .Input("type_emb2b_parameters: " STAF_TF_DTYPE)
    .Input("type_map: int32")
    .Output("two_body_afs: " STAF_TF_DTYPE);


void radialAFs_Launcher(const real* radial_descriptor,const int nr,const real* alpha2b_parameters,
        const int nalpha_r,real* radial_AFs,const int dimbat,const int N_local,
        const int* interaction_map_rad,const real* type_emb2b,const int* type_map, cudaStream_t stream);
void set_tensor_to_zero_real(real* tensor,int dimten, cudaStream_t stream);

class ComputeSortProjOp : public OpKernel {
 public:
  explicit ComputeSortProjOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const cudaStream_t stream = context->eigen_device<Eigen::GpuDevice>().stream();

    // Grab the input tensor
    const Tensor& radiale_T = context->input(0);
    const Tensor& intmap2b_T = context->input(1);
    const Tensor& alpha_radiale_T = context->input(2);
    const Tensor& type_emb2b_T = context->input(3);
    const Tensor& type_map_T = context->input(4);


    //flattizzo
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


    // Create an output tensor
    Tensor* radial_AFs_T = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (dimbat);
    grad_net_shape.AddDim (Nlocal);
    grad_net_shape.AddDim (nalpha_r);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &radial_AFs_T));

    //It seems tensorflow does not set to zero the pointed memory!
    set_tensor_to_zero_real(radial_AFs_T->flat<real>().data(),dimbat*Nlocal*nalpha_r, stream);

    //Calcolo della proiezione su base
    radialAFs_Launcher(
          radial_descriptor.data(),nr,alpha2b_parameters.data(),
          nalpha_r,radial_AFs_T->flat<real>().data(),dimbat,Nlocal,
          interaction_map_rad.data(),type_emb2b.data(),type_map.data()
    , stream);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeSortProj").Device(DEVICE_GPU), ComputeSortProjOp);
