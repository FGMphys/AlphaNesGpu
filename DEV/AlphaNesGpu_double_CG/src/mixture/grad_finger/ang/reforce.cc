#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("ComputeSortProj3bodyGrad")
    .Input("previous_gradient: double")
    .Input("angular_desciptors: double")
    .Input("radial_descriptors: double")
    .Input("intmap3b: int32")
    .Input("intmap2b: int32")
    .Input("alpha3b_parameters: double")
    .Input("type_emb3b: double")
    .Input("color_type_map: int32")
    .Input("num_triplets: int32")
    .Input("map_color_interaction: int32")
    .Input("map_intra: int32")
    .Output("alphagrad3body: double")
    .Output("nextgrad_emb3b: double");


void alphagrad_ang_Launcher(const double* radial_descriptor,const double* angular_descriptor,
                 int nr,int na,const double* prevgrad,int dimbat,
                 int Nlocal,const int* intmap3b,const double* alpha3b,
                 int nsmooth_a,double* next_alpha3b_grad,
                 const double* type_emb3b,const int* color_type_map,
                 double* next_emb3b_grad, const int* num_triplet,int nt_couple,
                 const int* map_color_interaction,const int* map_intra);

void set_tensor_to_zero_double(double* tensor,int dimten);


class ComputeSortProj3bodyGradOp : public OpKernel {
 public:
  explicit ComputeSortProj3bodyGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& prevgrad_T = context->input(0);
    const Tensor& angular_descriptor_T = context->input(1);
    const Tensor& radial_descriptor_T = context->input(2);
    const Tensor& interaction_map_angular_T = context->input(3);
    const Tensor& interaction_map_rad_T = context->input(4);
    const Tensor& alpha3b_parameters_T = context->input(5);
    const Tensor& type_emb3b_parameters_T = context->input(6);
    const Tensor& color_type_map_T = context->input(7);
    const Tensor& num_triplet_T = context->input(8);

    const Tensor& map_color_interaction_T = context->input(9);
    const Tensor& map_intra_T = context->input(10);

    auto prevgrad=prevgrad_T.flat<double>();
    auto angular_descriptor =  angular_descriptor_T.flat<double>();
    auto radial_descriptor = radial_descriptor_T.flat<double>();
    auto intmap3b = interaction_map_angular_T.flat<int>();
    auto interaction_map_rad = interaction_map_rad_T.flat<int>();
    auto alpha3b = alpha3b_parameters_T.flat<double>();
    auto type_emb3b = type_emb3b_parameters_T.flat<double>();
    auto color_type_map = color_type_map_T.flat<int>();
    auto num_triplet = num_triplet_T.flat<int>();

    auto map_color_interaction=map_color_interaction_T.flat<int>();
    auto map_intra=map_intra_T.flat<int>();

    //Prendo le dimensioni del tensore
    int dimbat = radial_descriptor_T.shape().dim_size(0);
    int nr = radial_descriptor_T.shape().dim_size(2);
    int na = angular_descriptor_T.shape().dim_size(2);
    int Nlocal = radial_descriptor_T.shape().dim_size(1);
    int nsmooth_a=int(alpha3b_parameters_T.shape().dim_size(1)/3);
    int nt_couple=alpha3b_parameters_T.shape().dim_size(0);

    // Create an output tensor for gradient wrt beta gamma delta AFs
    Tensor* next_alpha3b_grad_T = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (nt_couple);
    grad_net_shape.AddDim (nsmooth_a*3);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &next_alpha3b_grad_T));
    set_tensor_to_zero_double(next_alpha3b_grad_T->flat<double>().data(),nt_couple*nsmooth_a*3);

    // Create an output for gradient wrt embedding 3b
    Tensor* next_emb3b_grad_T = NULL;
    TensorShape grad_net_shape2 ;
    grad_net_shape2.AddDim (nt_couple);
    grad_net_shape2.AddDim (nsmooth_a);
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_net_shape2,
                                                     &next_emb3b_grad_T));
    set_tensor_to_zero_double(next_emb3b_grad_T->flat<double>().data(),nt_couple*nsmooth_a);

    alphagrad_ang_Launcher(radial_descriptor.data(),angular_descriptor.data(),
                     nr,na,prevgrad.data(),dimbat,
                     Nlocal,intmap3b.data(),alpha3b.data(),nsmooth_a,
                     next_alpha3b_grad_T->flat<double>().data(),
                     type_emb3b.data(),color_type_map.data(),
                     next_emb3b_grad_T->flat<double>().data(),num_triplet.data(),nt_couple,
                     map_color_interaction.data(),map_intra.data());
}

};

REGISTER_KERNEL_BUILDER(Name("ComputeSortProj3bodyGrad").Device(DEVICE_GPU), ComputeSortProj3bodyGradOp);
