#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

void init_block_dim(int buffdim);

REGISTER_OP("InitForceRadial")
    .Input("buffdim: int32")
    .Output("code: int32");

    class InitForceRadialOp : public OpKernel {
     public:
      explicit InitForceRadialOp(OpKernelConstruction* context) : OpKernel(context) {}
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
      REGISTER_KERNEL_BUILDER(Name("InitForceRadial").Device(DEVICE_CPU), InitForceRadialOp);


REGISTER_OP("ComputeForceRadial")
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
    .Output("force: double");



void computeforce_doublets_Launcher(const double*  netderiv, const double* des_r,
                    const double* intderiv_r,const int* intmap_r,
                    int nr, int N, int dimbat,int num_alpha_radiale,
                    const double* alpha_radiale,const double* type_emb2b,
                    const int* actual_type,double* forces2b,const int* color_type_map,
                    int prod, const int* map_color_interaction, const int* map_intra);


void set_tensor_to_zero_double(double* tensor,int dimten);

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
    const Tensor& color_type_map_T = context->input(6);

    const Tensor& map_color_interaction_T = context->input(7);
    const Tensor& actual_type_T = context->input(8);
    const Tensor& map_intra_T = context->input(9);


    //Grabbing some useful dimension
    int dimbat = netderiv_T.shape().dim_size(0);
    int nr = desr_T.shape().dim_size(2);
    int Nlocal=desr_T.shape().dim_size(1);
    int N = type_map_T.shape().dim_size(0);
    int num_alpha_radiale=alpha_radiale_T.shape().dim_size(1);

    //Getting data pointer
    auto netderiv_T_flat = netderiv_T.flat<double>();
    auto desder_T_flat = desder_T.flat<double>();
    auto intmap2b_T_flat = intmap2b_T.flat<int>();
    auto desr_T_flat = desr_T.flat<double>();
    auto alpha_radiale_T_flat = alpha_radiale_T.flat<double>();

    auto type_emb2b_T_flat = type_emb2b_T.flat<double>();
    auto color_type_map_T_flat = color_type_map_T.flat<int>();

    auto map_color_interaction_T_flat = map_color_interaction_T.flat<int>();
    auto map_intra_T_flat = map_intra_T.flat<int>();


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

    set_tensor_to_zero_double(forces2b_T->flat<double>().data(),dimbat*3*N);
    int prod=dimbat*Nlocal*nr;
   computeforce_doublets_Launcher(netderiv_T_flat.data(),desr_T_flat.data(),
   desder_T_flat.data(),intmap2b_T_flat.data(),nr,
   N,dimbat,num_alpha_radiale,alpha_radiale_T_flat.data(),
   type_emb2b_T_flat.data(),actual_type,
   forces2b_T->flat<double>().data(),color_type_map_T_flat.data(),prod,
   map_color_interaction_T_flat.data(),map_intra_T_flat.data());

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceRadial").Device(DEVICE_GPU), ComputeForceRadialOp);
