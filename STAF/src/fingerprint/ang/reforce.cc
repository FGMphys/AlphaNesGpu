#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "staf_real.h"

#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"





using namespace tensorflow;

REGISTER_OP("ComputeSortProj3body")
    .Input("angular_descriptor: " STAF_TF_DTYPE)
    .Input("radial_descriptor: " STAF_TF_DTYPE)
    .Input("interaction_map_angular: int32")
    .Input("interaction_map_rad: int32")
    .Input("alpha3b_parameters: " STAF_TF_DTYPE)
    .Input("type_emb3b_parameters: " STAF_TF_DTYPE)
    .Input("type_map: int32")
    .Input("num_triplet: int32")
    .Output("three_body_afs: " STAF_TF_DTYPE);

void angularAFs_Launcher(const real* radial_descriptor,const real* angular_descriptor,int nr,int na,
                          real* three_body_AFs,int dimbat,int Nlocal,
                          const int* interaction_map_angular,const real* alpha3b_parameters,
                          int nsmooth_a,const real* type_emb3b,
                          const int* type_map,const int* num_triplets, cudaStream_t stream);

void set_tensor_to_zero_real(real* tensor,int dimten, cudaStream_t stream);

class ComputeSortProj3bodyOp : public OpKernel {
 public:
  explicit ComputeSortProj3bodyOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const cudaStream_t stream = context->eigen_device<Eigen::GpuDevice>().stream();

    // Grab the input tensor
    const Tensor& angular_descriptor_T = context->input(0);
    const Tensor& radial_descriptor_T = context->input(1);
    const Tensor& interaction_map_angular_T = context->input(2);
    const Tensor& interaction_map_rad_T = context->input(3);
    const Tensor& alpha3b_parameters_T = context->input(4);

    const Tensor& type_emb3b_parameters_T = context->input(5);
    const Tensor& type_map_T = context->input(6);

    const Tensor& num_triplet_T = context->input(7);

    //flattizzo
    auto angular_descriptor =  angular_descriptor_T.flat<real>();
    auto radial_descriptor = radial_descriptor_T.flat<real>();
    auto interaction_map_angular = interaction_map_angular_T.flat<int>();
    auto interaction_map_rad = interaction_map_rad_T.flat<int>();
    auto alpha3b_parameters = alpha3b_parameters_T.flat<real>();

    auto type_emb3b = type_emb3b_parameters_T.flat<real>();
    auto type_map = type_map_T.flat<int>();

    auto num_triplet = num_triplet_T.flat<int>();

    //Prendo le dimensioni del tensore
    int dimbat = radial_descriptor_T.shape().dim_size(0);

    int nr = radial_descriptor_T.shape().dim_size(2);

    int na = angular_descriptor_T.shape().dim_size(2);

    int Nlocal = radial_descriptor_T.shape().dim_size(1);

    int nsmooth_a=int(alpha3b_parameters_T.shape().dim_size(1)/3);

    // Create an output tensor
    Tensor* angular_AFs_T = NULL;
    TensorShape angular_AFs_shape ;
    angular_AFs_shape.AddDim (dimbat);
    angular_AFs_shape.AddDim (Nlocal);
    angular_AFs_shape.AddDim (nsmooth_a);
    OP_REQUIRES_OK(context, context->allocate_output(0, angular_AFs_shape,
                                                     &angular_AFs_T));
    set_tensor_to_zero_real(angular_AFs_T->flat<real>().data(),dimbat*Nlocal*nsmooth_a, stream);

    //Computing three-body atomic-fingerprints
    angularAFs_Launcher(radial_descriptor.data(),angular_descriptor.data(),nr,na,
                          angular_AFs_T->flat<real>().data(),dimbat,Nlocal,interaction_map_angular.data(),
                          alpha3b_parameters.data(),nsmooth_a,type_emb3b.data(),
                          type_map.data(),num_triplet.data(), stream);
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeSortProj3body").Device(DEVICE_GPU), ComputeSortProj3bodyOp);
