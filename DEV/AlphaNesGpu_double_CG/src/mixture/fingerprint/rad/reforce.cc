#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"


using namespace tensorflow;

REGISTER_OP("ComputeSortProj")
    .Input("radial_descriptor: double")
    .Input("interaction_map_rad: int32")
    .Input("alpha2b_parameters: double")
    .Input("type_emb2b_parameters: double")
    .Input("color_type_map: int32")
    .Input("map_color_interaction: int32")
    .Input("map_intra: int32")
    .Output("two_body_afs: double");


void radialAFs_Launcher(const double* radial_descriptor,const int nr,const double* alpha2b_parameters,
        const int nalpha_r,double* radial_AFs,const int dimbat,const int N_local,
        const int* interaction_map_rad,const double* type_emb2b,const int* color_type_map,
        const int* map_color_interaction,const int* map_intra);
void set_tensor_to_zero_double(double* tensor,int dimten);

class ComputeSortProjOp : public OpKernel {
 public:
  explicit ComputeSortProjOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& radiale_T = context->input(0);
    const Tensor& intmap2b_T = context->input(1);
    const Tensor& alpha_radiale_T = context->input(2);
    const Tensor& type_emb2b_T = context->input(3);
    const Tensor& color_type_map_T = context->input(4);
    const Tensor& map_color_interaction_T = context->input(5);
    const Tensor& map_intra_T = context->input(6);

    //flattizzo
    auto radial_descriptor = radiale_T.flat<double>();
    auto interaction_map_rad = intmap2b_T.flat<int>();
    auto alpha2b_parameters = alpha_radiale_T.flat<double>();

    auto type_emb2b = type_emb2b_T.flat<double>();
    auto color_type_map = type_map_T.flat<int>();
    auto map_color_interaction = map_color_interaction_T.flat<int>();
    auto map_intra = map_intra_T.flat<int>();


    //Prendo le dimensioni del tensore
    int dimbat = radiale_T.shape().dim_size(0);
    int nr = radiale_T.shape().dim_size(2);
    int Nlocal = radiale_T.shape().dim_size(1);
    int nalpha_r= alpha_radiale_T.shape().dim_size(1);


    // Create an output tensor
    Tensor* radial_AFs_T = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (dimbat);
    grad_net_shape.AddDim (Nlocal);
    grad_net_shape.AddDim (nalpha_r);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &radial_AFs_T));

    //It seems tensorflow does not set to zero the pointed memory!
    set_tensor_to_zero_double(radial_AFs_T->flat<double>().data(),dimbat*Nlocal*nalpha_r);

    //Calcolo della proiezione su base
    radialAFs_Launcher(
          radial_descriptor.data(),nr,alpha2b_parameters.data(),
          nalpha_r,radial_AFs_T->flat<double>().data(),dimbat,Nlocal,
          interaction_map_rad.data(),type_emb2b.data(),color_type_map.data(),
          map_color_interaction.data(),map_intra.data());

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeSortProj").Device(DEVICE_GPU), ComputeSortProjOp);
