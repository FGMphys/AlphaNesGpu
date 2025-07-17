#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
//#include <cuda.h>
using namespace tensorflow;

void init_block_dim(int buffdim);

REGISTER_OP("InitForceTripl")
    .Input("buffdim: int32")
    .Output("code: int32");

    class InitForceTriplOp : public OpKernel {
     public:
      explicit InitForceTriplOp(OpKernelConstruction* context) : OpKernel(context) {}
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
      REGISTER_KERNEL_BUILDER(Name("InitForceTripl").Device(DEVICE_CPU), InitForceTriplOp);






REGISTER_OP("ComputeForceTripl")
    .Input("netderiv: double")
    .Input("radial_descriptor: double")
    .Input("angular_descriptor: double")
    .Input("descriptor_derivative_rad: double")
    .Input("descriptor_derivative_ang: double")
    .Input("interaction_map_rad: int32")
    .Input("interaction_map_ang: int32")
    .Input("alpha3b_parameters: double")
    .Input("type_emb3b_parameters: double")
    .Input("color_type_map: int32")
    .Input("map_color_interaction: int32")
    .Input("actual_type: int32")
    .Input("num_triplets: int32")
    .Input("map_intra: int32")
    .Output("force: double");

    void computeforce_tripl_Launcher(const double*  netderiv_T_d, const double* desr_T_d, const double* desa_T_d,
                        const double* intderiv_r_T_d, const double* intderiv_a_T_d,
                        const int* intmap_r_T_d,const int* intmap_a_T_d,
                        int nr, int na, int N, int dimbat,int num_finger,const double* type_emb3b_d,
                        const int* actual_type,double* forces3b_T_d,const int *num_triplets_d,
                        const double* smooth_a_T,const int* color_type_map_T_d,
                        int prod, const int* map_color_interaction, const int* map_intra);

void set_tensor_to_zero_double(double* tensor,int dimten);

class ComputeForceTriplOp : public OpKernel {
 public:
  explicit ComputeForceTriplOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& netderiv_T = context->input(0);
    const Tensor& desr_T = context->input(1);
    const Tensor& desa_T = context->input(2);
    const Tensor& intderiv_r_T = context->input(3);
    const Tensor& intderiv_a_T = context->input(4);
    const Tensor& intmap_r_T = context->input(5);
    const Tensor& intmap_a_T = context->input(6);
    const Tensor& smooth_a_T = context->input(7);

    const Tensor& type_emb3b_T = context->input(8);
    const Tensor& color_type_map_T = context->input(9);

    const Tensor& actual_type_T = context->input(11);

    const Tensor& num_triplets_T=context->input(12);

    const Tensor& map_color_interaction_T=context->input(10);
    const Tensor& map_intra_T=context->input(13);

    //flatting the tensor
    int dimbat = netderiv_T.shape().dim_size(0);
    //int dimbat=dimbat_flat(0);

    int nr = desr_T.shape().dim_size(2);
    //int nr=numdes2body_flat(0);

    int na = desa_T.shape().dim_size(2);
    //int na=numdes3body_flat(0);

    int N = color_type_map_T.shape().dim_size(0);
    //int N=N_flat(0);

    const int* actual_type=actual_type_T.flat<int>().data();
    //int actual_type=actual_type_T_flat(0);

    int num_finger=int(smooth_a_T.shape().dim_size(1)/3);

    //printf("\n\n %d %d %d %d %d %d  \n\n",dimbat,nr,na,N,nt,num_finger);


    //Arrays read
    auto netderiv_T_d=netderiv_T.flat<double>();
    auto desr_T_d= desr_T.flat<double>();
    auto desa_T_d= desa_T.flat<double>();
    auto intderiv_r_T_d=intderiv_r_T.flat<double>();
    auto intderiv_a_T_d=intderiv_a_T.flat<double>();
    auto intmap_r_T_d=intmap_r_T.flat<int>();
    auto intmap_a_T_d=intmap_a_T.flat<int>();
    auto type_emb3b_T_d=type_emb3b_T.flat<double>();
    auto smooth_a_T_d=smooth_a_T.flat<double>();
    auto color_type_map_T_d=color_type_map_T.flat<int>();
    auto num_triplets_T_d=num_triplets_T.flat<int>();
    auto map_color_interaction_T_d=map_color_interaction_T.flat<int>();
    auto map_intra_T_d=map_intra_T.flat<int>();
    // Create an output tensor for forces
    Tensor* forces3b_T = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (dimbat);
    grad_net_shape.AddDim (N*3);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &forces3b_T));

    set_tensor_to_zero_double(forces3b_T->flat<double>().data(),dimbat*3*N);
    int prod=netderiv_T.shape().dim_size(0)*netderiv_T.shape().dim_size(1)*desa_T.shape().dim_size(2);//dimbat*Nlocal*na
    computeforce_tripl_Launcher(netderiv_T_d.data(), desr_T_d.data(), desa_T_d.data(),
                        intderiv_r_T_d.data(),intderiv_a_T_d.data(),
                        intmap_r_T_d.data(),intmap_a_T_d.data(),
                        nr, na, N, dimbat,num_finger,type_emb3b_T_d.data(),
                        actual_type,forces3b_T->flat<double>().data(),
                        num_triplets_T_d.data(),smooth_a_T_d.data(),
                        color_type_map_T_d.data(),prod,map_color_interaction_T_d.data(),
                        map_intra_T_d.data());

}
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceTripl").Device(DEVICE_GPU), ComputeForceTriplOp);
