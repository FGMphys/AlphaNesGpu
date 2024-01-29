#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
//#include <cuda.h>
using namespace tensorflow;


REGISTER_OP("ComputeForceTripl")
    .Input("netderiv: float")
    .Input("radial_descriptor: float")
    .Input("angular_descriptor: float")
    .Input("descriptor_derivative_rad: float")
    .Input("descriptor_derivative_ang: float")
    .Input("interaction_map_rad: int32")
    .Input("interaction_map_ang: int32")
    .Input("alpha3b_parameters: float")
    .Input("type_emb3b_parameters: float")
    .Input("type_map: int32")
    .Input("tipos: int32")
    .Input("actual_type: int32")
    .Input("num_triplets: int32")
    .Output("force: float");

    void computeforce_tripl_Launcher(const float*  netderiv_T_d, const float* desr_T_d, const float* desa_T_d,
                        const float* intderiv_r_T_d, const float* intderiv_a_T_d,
                        const int* intmap_r_T_d,const int* intmap_a_T_d,
                        int nr, int na, int N, int dimbat,int num_finger,const float* type_emb3b_d,int nt,
                        const int* tipos_T,
                        const int* actual_type,float* forces3b_T_d,const int *num_triplets_d,const float* smooth_a_T,const int* type_map_T_d,int prod);

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
    const Tensor& type_map_T = context->input(9);

    const Tensor& tipos_T = context->input(10);
    const Tensor& actual_type_T = context->input(11);
    
    const Tensor& num_triplets_T=context->input(12);

    //flatting the tensor
    int dimbat = netderiv_T.shape().dim_size(0);
    //int dimbat=dimbat_flat(0);

    int nr = desr_T.shape().dim_size(2);
    //int nr=numdes2body_flat(0);

    int na = desa_T.shape().dim_size(2);
    //int na=numdes3body_flat(0);

    int N = type_map_T.shape().dim_size(0);
    //int N=N_flat(0);

    int nt = tipos_T.shape().dim_size(0);
    
    const int* actual_type=actual_type_T.flat<int>().data();
    //int actual_type=actual_type_T_flat(0);

    int num_finger=smooth_a_T.shape().dim_size(1);
    //int num_finger=number_alpha3b_T_flat(0);

    //printf("\n\n %d %d %d %d %d %d  \n\n",dimbat,nr,na,N,nt,num_finger);
   

    //Arrays read
    auto netderiv_T_d=netderiv_T.flat<float>();
    auto desr_T_d= desr_T.flat<float>();
    auto desa_T_d= desa_T.flat<float>();
    auto intderiv_r_T_d=intderiv_r_T.flat<float>();
    auto intderiv_a_T_d=intderiv_a_T.flat<float>();
    auto intmap_r_T_d=intmap_r_T.flat<int>();
    auto intmap_a_T_d=intmap_a_T.flat<int>();
    auto type_emb3b_T_d=type_emb3b_T.flat<float>();
    auto smooth_a_T_d=smooth_a_T.flat<float>();
    auto type_map_T_d=type_map_T.flat<int>();
    auto tipos_T_d=tipos_T.flat<int>();
    auto num_triplets_T_d=num_triplets_T.flat<int>();
    // Create an output tensor for forces
    Tensor* forces3b_T = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (dimbat);
    grad_net_shape.AddDim (N);
    grad_net_shape.AddDim (4);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &forces3b_T));

    
    int prod=netderiv_T.shape().dim_size(0)*netderiv_T.shape().dim_size(1)*desa_T.shape().dim_size(2);//dimbat*Nlocal*na
    //COMPUTING FORCES
   //     if (actual_type_T.flat<int>()(0)==0)
 //     printf("\n%f\n",forces3b_T->flat<float>()(0));

    computeforce_tripl_Launcher(netderiv_T_d.data(), desr_T_d.data(), desa_T_d.data(),
                        intderiv_r_T_d.data(),intderiv_a_T_d.data(),
                        intmap_r_T_d.data(),intmap_a_T_d.data(),
                        nr, na, N, dimbat,num_finger,type_emb3b_T_d.data(),nt,
                        tipos_T_d.data(),
                        actual_type,forces3b_T->flat<float>().data(),num_triplets_T_d.data(),smooth_a_T_d.data(),type_map_T_d.data(),prod);
 
}
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceTripl").Device(DEVICE_GPU), ComputeForceTriplOp);
