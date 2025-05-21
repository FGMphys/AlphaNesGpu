#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"




using namespace tensorflow;

REGISTER_OP("ComputeSortProj3body")
    .Input("angular_descriptor: double")
    .Input("radial_descriptor: double")
    .Input("interaction_map_angular: int32")
    .Input("interaction_map_rad: int32")
    .Input("alpha3b_parameters: double")
    .Input("type_emb3b_parameters: double")
    .Input("type_map: int32")
    .Input("num_triplet: int32")
    .Output("three_body_afs: double");

void angularAFs_Launcher(const double* radial_descriptor,const double* angular_descriptor,int nr,int na,
                          double* three_body_AFs,int dimbat,int Nlocal,
                          const int* interaction_map_angular,const double* alpha3b_parameters,
                          int nsmooth_a,const double* type_emb3b,
                          const int* type_map,const int* num_triplets);

void set_tensor_to_zero_double(double* tensor,int dimten);

class ComputeSortProj3bodyOp : public OpKernel {
 public:
  explicit ComputeSortProj3bodyOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
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
    auto angular_descriptor =  angular_descriptor_T.flat<double>();
    auto radial_descriptor = radial_descriptor_T.flat<double>();
    auto interaction_map_angular = interaction_map_angular_T.flat<int>();
    auto interaction_map_rad = interaction_map_rad_T.flat<int>();
    auto alpha3b_parameters = alpha3b_parameters_T.flat<double>();

    auto type_emb3b = type_emb3b_parameters_T.flat<double>();
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
    set_tensor_to_zero_double(angular_AFs_T->flat<double>().data(),dimbat*Nlocal*nsmooth_a);

    //Computing three-body atomic-fingerprints
    angularAFs_Launcher(radial_descriptor.data(),angular_descriptor.data(),nr,na,
                          angular_AFs_T->flat<double>().data(),dimbat,Nlocal,interaction_map_angular.data(),
                          alpha3b_parameters.data(),nsmooth_a,type_emb3b.data(),
                          type_map.data(),num_triplet.data());
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeSortProj3body").Device(DEVICE_GPU), ComputeSortProj3bodyOp);
