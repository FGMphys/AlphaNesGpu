#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "staf_real.h"

#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"

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
    .Input("netderiv: " STAF_TF_DTYPE)
    .Input("descriptor_derivative_rad: " STAF_TF_DTYPE)
    .Input("interaction_map_rad: int32")
    .Input("radial_descriptor: " STAF_TF_DTYPE)
    .Input("alpha2b_parameters: " STAF_TF_DTYPE)
    .Input("type_emb2b_parameters: " STAF_TF_DTYPE)
    .Input("type_map: int32")
    .Input("tipos: int32")
    .Input("actual_type: int32")
    .Output("force: " STAF_TF_DTYPE);



void computeforce_doublets_Launcher(const real*  netderiv, const real* des_r,
                    const real* intderiv_r,const int* intmap_r,
                    int nr, int N, int dimbat,int num_alpha_radiale,
                    const real* alpha_radiale,const real* type_emb2b,int nt,
                    const int* tipos_T,const int* actual_type,real* forces2b,const int* type_map,int prod, cudaStream_t stream);


void set_tensor_to_zero_real(real* tensor,int dimten, cudaStream_t stream);

class ComputeForceRadialOp : public OpKernel {
 public:
  explicit ComputeForceRadialOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const cudaStream_t stream = context->eigen_device<Eigen::GpuDevice>().stream();

    // Grab the input tensor
    const Tensor& netderiv_T = context->input(0);
    const Tensor& desder_T = context->input(1);
    const Tensor& intmap2b_T = context->input(2);
    const Tensor& desr_T = context->input(3);
    const Tensor& alpha_radiale_T = context->input(4);

    const Tensor& type_emb2b_T = context->input(5);
    const Tensor& type_map_T = context->input(6);

    const Tensor& tipos_T = context->input(7);
    const Tensor& actual_type_T = context->input(8);


    //Grabbing some useful dimension
    int dimbat = netderiv_T.shape().dim_size(0);
    int nr = desr_T.shape().dim_size(2);
    int Nlocal=desr_T.shape().dim_size(1);
    int N = type_map_T.shape().dim_size(0);
    int nt = tipos_T.shape().dim_size(0);
    int num_alpha_radiale=alpha_radiale_T.shape().dim_size(1);

    //Getting data pointer
    auto netderiv_T_flat = netderiv_T.flat<real>();
    auto desder_T_flat = desder_T.flat<real>();
    auto intmap2b_T_flat = intmap2b_T.flat<int>();
    auto desr_T_flat = desr_T.flat<real>();
    auto alpha_radiale_T_flat = alpha_radiale_T.flat<real>();

    auto type_emb2b_T_flat = type_emb2b_T.flat<real>();
    auto type_map_T_flat = type_map_T.flat<int>();

    auto tipos_T_flat = tipos_T.flat<int>();

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

    set_tensor_to_zero_real(forces2b_T->flat<real>().data(),dimbat*3*N, stream);
    int prod=dimbat*Nlocal*nr;
   computeforce_doublets_Launcher(netderiv_T_flat.data(),desr_T_flat.data(),desder_T_flat.data(),intmap2b_T_flat.data(),nr,N,dimbat,num_alpha_radiale,alpha_radiale_T_flat.data(),type_emb2b_T_flat.data(),nt,tipos_T_flat.data(),actual_type,forces2b_T->flat<real>().data(),type_map_T_flat.data(),prod, stream);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceRadial").Device(DEVICE_GPU), ComputeForceRadialOp);

void computeforce_doublets_virial_Launcher(
                    const real*  netderiv, const real* des_r,
                    const real* intderiv_r,const int* intmap_r,
                    int nr, int N, int dimbat,int num_alpha_radiale,
                    const real* alpha_radiale,const real* type_emb2b,int nt,
                    const int* tipos_T,const int* actual_type,real* forces2b,const int* type_map,int prod,
                    real* virial_diagonal_d,const real* pos_d,const real* box_d,
                    cudaStream_t stream);

REGISTER_OP("ComputeForceRadialVirial")
    .Input("netderiv: " STAF_TF_DTYPE)
    .Input("descriptor_derivative_rad: " STAF_TF_DTYPE)
    .Input("interaction_map_rad: int32")
    .Input("radial_descriptor: " STAF_TF_DTYPE)
    .Input("alpha2b_parameters: " STAF_TF_DTYPE)
    .Input("type_emb2b_parameters: " STAF_TF_DTYPE)
    .Input("type_map: int32")
    .Input("tipos: int32")
    .Input("actual_type: int32")
    .Input("pos: " STAF_TF_DTYPE)
    .Input("box: " STAF_TF_DTYPE)
    .Output("force: " STAF_TF_DTYPE)
    .Output("virial_diag: " STAF_TF_DTYPE);

class ComputeForceRadialVirialOp : public OpKernel {
 public:
  explicit ComputeForceRadialVirialOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const cudaStream_t stream = context->eigen_device<Eigen::GpuDevice>().stream();

    const Tensor& netderiv_T = context->input(0);
    const Tensor& desder_T = context->input(1);
    const Tensor& intmap2b_T = context->input(2);
    const Tensor& desr_T = context->input(3);
    const Tensor& alpha_radiale_T = context->input(4);
    const Tensor& type_emb2b_T = context->input(5);
    const Tensor& type_map_T = context->input(6);
    const Tensor& tipos_T = context->input(7);
    const Tensor& actual_type_T = context->input(8);
    const Tensor& pos_T = context->input(9);
    const Tensor& box_T = context->input(10);

    int dimbat = netderiv_T.shape().dim_size(0);
    int nr = desr_T.shape().dim_size(2);
    int Nlocal=desr_T.shape().dim_size(1);
    int N = type_map_T.shape().dim_size(0);
    int nt = tipos_T.shape().dim_size(0);
    int num_alpha_radiale=alpha_radiale_T.shape().dim_size(1);

    const int* actual_type=actual_type_T.flat<int>().data();

    Tensor* forces2b_T = NULL;
    TensorShape force_shape;
    force_shape.AddDim(dimbat);
    force_shape.AddDim(N*3);
    OP_REQUIRES_OK(context, context->allocate_output(0, force_shape, &forces2b_T));

    Tensor* virial_T = NULL;
    TensorShape virial_shape;
    virial_shape.AddDim(dimbat);
    virial_shape.AddDim(3);
    OP_REQUIRES_OK(context, context->allocate_output(1, virial_shape, &virial_T));

    set_tensor_to_zero_real(forces2b_T->flat<real>().data(), dimbat*3*N, stream);
    set_tensor_to_zero_real(virial_T->flat<real>().data(), dimbat*3, stream);

    int prod=dimbat*Nlocal*nr;
    computeforce_doublets_virial_Launcher(
        netderiv_T.flat<real>().data(), desr_T.flat<real>().data(),
        desder_T.flat<real>().data(), intmap2b_T.flat<int>().data(),
        nr, N, dimbat, num_alpha_radiale,
        alpha_radiale_T.flat<real>().data(), type_emb2b_T.flat<real>().data(), nt,
        tipos_T.flat<int>().data(), actual_type,
        forces2b_T->flat<real>().data(), type_map_T.flat<int>().data(), prod,
        virial_T->flat<real>().data(), pos_T.flat<real>().data(), box_T.flat<real>().data(),
        stream);
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceRadialVirial").Device(DEVICE_GPU),
                        ComputeForceRadialVirialOp);

