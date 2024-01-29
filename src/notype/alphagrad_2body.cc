///Implementazione del gradiente di una funzione scalare L(SD), funzione dei SD(alpha).


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"


using namespace tensorflow;
void compute_2bodyalphagrad(const Tensor& descriptors_T,int numdes,const Tensor& alpha_radiale_T,
                            int nalpha_r,Tensor* next_alpah2b_grad_T, int dimbat,
                            int N, const Tensor& intmap2b_T, const Tensor& prev_grad_T)
{
  double prevgradel;
//Making easy accessible tensorflow input and OUTPUT
auto numneigh=intmap2b_T.flat<int>();
auto ds=descriptors_T.flat<double>();
auto prevgrad=prev_grad_T.flat<double>();
auto alpha=alpha_radiale_T.flat<double>();

auto nextgrad=next_alpah2b_grad_T->flat<double>();

for (int k=0;k<nalpha_r;k++){
    nextgrad(k)=0.;
}


for (int b=0;b<dimbat;b++){

    for (int par=0;par<N;par++){
        int num=numneigh(b*N*(numdes+1)+par*(numdes+1));
        int actual=b*N*numdes+par*numdes;
        int j;
        for (int j=0;j<num;j++)
        {
          for (int i=0;i<nalpha_r;i++){
              double ds_el=ds(actual+j);
              prevgradel=prevgrad(b*nalpha_r*N+par*nalpha_r+i);
              nextgrad(i)+=ds_el*ds_el*exp(alpha(i)*ds_el)*prevgradel;
             }
          }
           }
         }
       }












using namespace tensorflow;

REGISTER_OP("ComputeTwoBodyParGrad")
    .Input("prev_grad: double")
    .Input("radial_descriptor: double")
    .Input("number_of_particles: int32")
    .Input("radial_descriptor_buffsize: int32")
    .Input("batch_dimension: int32")
    .Input("interaction_map_rad: int32")
    .Input("alpha2b_parameters: double")
    .Input("number_alpha2b: int32")
    .Output("nextgrad_alpha2b: double");





class ComputeTwoBodyParGradOp : public OpKernel {
 public:
  explicit ComputeTwoBodyParGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& prev_grad_T = context->input(0);
    const Tensor& radiale_T = context->input(1);
    const Tensor& numpar_T = context->input(2);
    const Tensor& des_buffsize_T = context->input(3);
    const Tensor& dimbatch_T = context->input(4);
    const Tensor& intmap2b_T = context->input(5);
    const Tensor& alpha_radiale_T = context->input(6);
    const Tensor& num_alpha2b_T = context->input(7);



    auto  dimbatch_T_flat = dimbatch_T.flat<int>();
    auto des_buffsize_T_flat = des_buffsize_T.flat<int>();
    auto numpar_T_flat = numpar_T.flat<int>();
    auto num_alpha2b_T_flat = num_alpha2b_T.flat<int>();

    //Tensor Dimension Variables
    int dimbat=dimbatch_T_flat(0);
    int numdes=des_buffsize_T_flat(0);
    int N=numpar_T_flat(0);
    int dimdes=dimbat*numdes*N;
    int nalpha_r=num_alpha2b_T_flat(0);


    // Create an output tensor for BackProp of alphas
    Tensor* nextgrad_alpha2b_T = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (nalpha_r);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &nextgrad_alpha2b_T));


    compute_2bodyalphagrad(radiale_T,numdes,alpha_radiale_T,nalpha_r,
                           nextgrad_alpha2b_T,dimbat,N,intmap2b_T,
                           prev_grad_T);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeTwoBodyParGrad").Device(DEVICE_CPU), ComputeTwoBodyParGradOp);
