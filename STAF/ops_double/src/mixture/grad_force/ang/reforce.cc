#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

void init_block_dim(int buffdim);

REGISTER_OP("InitGradForceTripl")
    .Input("buffdim: int32")
    .Output("code: int32");

    class InitGradForceTriplOp : public OpKernel {
     public:
      explicit InitGradForceTriplOp(OpKernelConstruction* context) : OpKernel(context) {}
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
      REGISTER_KERNEL_BUILDER(Name("InitGradForceTripl").Device(DEVICE_CPU), InitGradForceTriplOp);

REGISTER_OP("ComputeForceTriplGrad")
    .Input("prevgrad: double")
    .Input("netderiv: double")
    .Input("radial_descriptor: double")
    .Input("angular_descriptor: double")
    .Input("descriptor_derivative_rad: double")
    .Input("descriptor_derivative_ang: double")
    .Input("interaction_map_rad: int32")
    .Input("interaction_map_ang: int32")
    .Input("alpha3b_parameters: double")
    .Input("type_emb3b_parameters: double")
    .Input("type_map: int32")
    .Input("tipos: int32")
    .Input("actual_type: int32")
    .Input("num_triplets: int32")
    .Output("gradnet: double")
    .Output("gradalpha: double")
    .Output("gradck: double");

    void gradforce_tripl_Launcher(const double*  prevgrad_T_d,const double*  netderiv_T_d, const double* desr_T_d,
                                      const double* desa_T_d,const double* intderiv_r_T_d,
                                      const double* intderiv_a_T_d,const int* intmap_r_T_d,
                                      const int* intmap_a_T_d,int nr, int na, int N,
                                      int dimbat,int num_finger,const double* type_emb3b_d,int nt,
                                      const int* tipos_T,const int* actual_type,
                                      const int *num_triplets_d,const double* smooth_a_T,
                                      const int* type_map_T_d,int prod,double* gradnet_3b_T_d,
                                      double* grad_alpha3b_T_d,double* grad_emb3b_T_d);


void set_tensor_to_zero_double(double* tensor,int dim);

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
    int dimbat = desr_T.shape().dim_size(0);

    int nr = desr_T.shape().dim_size(2);

    int na = desa_T.shape().dim_size(2);

    int N = type_map_T.shape().dim_size(0);
    int Nlocal = desr_T.shape().dim_size(1);

    int nt = tipos_T.shape().dim_size(0);
    int nt_couple=nt*(nt+1)/2;

    const int* actual_type=actual_type_T.flat<int>().data();

    int num_finger=int(smooth_a_T.shape().dim_size(1)/3);

    //Flatting tensor to be used as arrays
    auto prevgrad_T_d=prevgrad_T.flat<double>();
    auto netderiv_T_d=netderiv_T.flat<double>();
    auto desr_T_d= desr_T.flat<double>();
    auto desa_T_d= desa_T.flat<double>();
    auto intderiv_r_T_d=intderiv_r_T.flat<double>();
    auto intderiv_a_T_d=intderiv_a_T.flat<double>();
    auto intmap_r_T_d=intmap_r_T.flat<int>();
    auto intmap_a_T_d=intmap_a_T.flat<int>();
    auto type_emb3b_T_d=type_emb3b_T.flat<double>();
    auto smooth_a_T_d=smooth_a_T.flat<double>();
    auto type_map_T_d=type_map_T.flat<int>();
    auto tipos_T_d=tipos_T.flat<int>();
    auto num_triplets_T_d=num_triplets_T.flat<int>();

    // Create an output tensor for DL/DT
    Tensor* gradnet_3b_T = NULL;
    TensorShape gradnet_3b_shape ;
//   gradnet_3b_shape.AddDim (1);
    gradnet_3b_shape.AddDim (dimbat);
    gradnet_3b_shape.AddDim (Nlocal);
    gradnet_3b_shape.AddDim (num_finger);
    OP_REQUIRES_OK(context, context->allocate_output(0, gradnet_3b_shape,
                                                     &gradnet_3b_T));
    set_tensor_to_zero_double(gradnet_3b_T->flat<double>().data(),dimbat*Nlocal*num_finger);
    // Create an output tensor for DL/Dalpha
    Tensor* grad_alpha3b_T = NULL;
    TensorShape grad_alpha3b_shape ;
    grad_alpha3b_shape.AddDim (nt_couple);
    grad_alpha3b_shape.AddDim (3*num_finger);
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_alpha3b_shape,
                                                     &grad_alpha3b_T));
    int dimnow=nt_couple*3*num_finger;
    set_tensor_to_zero_double(grad_alpha3b_T->flat<double>().data(),dimnow);
    // Create an output tensor for DL/Dck
    Tensor* grad_emb3b_T = NULL;
    TensorShape grad_emb3b_shape;
    grad_emb3b_shape.AddDim (nt_couple);
    grad_emb3b_shape.AddDim (num_finger);
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_emb3b_shape,
                                                     &grad_emb3b_T));
    set_tensor_to_zero_double(grad_emb3b_T->flat<double>().data(),nt_couple*num_finger);


    int prod=dimbat*Nlocal*na;
    gradforce_tripl_Launcher(prevgrad_T_d.data(),netderiv_T_d.data(), desr_T_d.data(), desa_T_d.data(),intderiv_r_T_d.data(),intderiv_a_T_d.data(),
                        intmap_r_T_d.data(),intmap_a_T_d.data(),
                        nr, na, N, dimbat,num_finger,type_emb3b_T_d.data(),nt,
                        tipos_T_d.data(),
                        actual_type,num_triplets_T_d.data(),smooth_a_T_d.data(),
                        type_map_T_d.data(),prod,gradnet_3b_T->flat<double>().data(),
                        grad_alpha3b_T->flat<double>().data(),
  			grad_emb3b_T->flat<double>().data());

}
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceTriplGrad").Device(DEVICE_GPU), ComputeForceTriplGradOp);
