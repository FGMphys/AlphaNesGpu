#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"


using namespace tensorflow;

void computeSmoothMax(const Tensor& descriptors_T,int num_ds,const Tensor& alpha_radiale_T,int nalpha_r,
                      Tensor* twobodyafs_T, int dimbat, int N, const Tensor& intmap2b_T,
                      const Tensor& type_emb2b_T,int nt, const Tensor& type_map_T)
{
  int par,b,i;
  int numdes=num_ds;

  //Making easy accessible tensorflow input and OUTPUT
  auto numneigh=intmap2b_T.flat<int>();
  auto type_map=type_map_T.flat<int>();
  auto type_emb2b=type_emb2b_T.flat<double>();
  auto ds=descriptors_T.flat<double>();
  auto alpha=alpha_radiale_T.flat<double>();
  auto twobodyafs=twobodyafs_T->flat<double>();

  int y=0;
  for (y=0;y<dimbat*N*nalpha_r;y++){
      twobodyafs(y)=0.;
  }

  // costruiamo i descrittori
for (b=0;b<dimbat;b++){
    for (par=0;par<N;par++){
        int num=numneigh(b*N*(numdes+1)+par*(numdes+1));
        int actual=b*N*numdes+par*numdes;
        //Proiezione su base e derivazione del singolo termine proiettato per la particella par nel frame b
        int j;
        for (j=0;j<num;j++){
            int neighj=numneigh(b*N*(numdes+1)+par*(numdes+1)+1+j);
            int ch_type=type_map(neighj);
        for (i=0;i<nalpha_r;i++)
            {
              double buffer1=ds(actual+j)*exp(alpha(nalpha_r*ch_type+i)*ds(actual+j))*type_emb2b(nalpha_r*ch_type+i);
              twobodyafs(b*nalpha_r*N+par*nalpha_r+i)+=buffer1;
             }
             }
         }
     }
}







using namespace tensorflow;

REGISTER_OP("ComputeSortProj")
    .Input("radial_descriptor: double")
    .Input("number_of_local_particles: int32")
    .Input("radial_descriptor_buffsize: int32")
    .Input("batch_dimension: int32")
    .Input("interaction_map_rad: int32")
    .Input("alpha2b_parameters: double")
    .Input("number_alpha2b: int32")
    .Input("type_emb2b_parameters: double")
    .Input("number_of_atom_type: int32")
    .Input("type_map: int32")
    .Output("two_body_afs: double");





class ComputeSortProjOp : public OpKernel {
 public:
  explicit ComputeSortProjOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& radiale_T = context->input(0);
    const Tensor& numpar_T = context->input(1);
    const Tensor& des_buffsize_T = context->input(2);
    const Tensor& dimbatch_T = context->input(3);
    const Tensor& intmap2b_T = context->input(4);
    const Tensor& alpha_radiale_T = context->input(5);
    const Tensor& num_alpha2b_T = context->input(6);
    const Tensor& type_emb2b_T = context->input(7);
    const Tensor& nt_T = context->input(8);
    const Tensor& type_map_T = context->input(9);


    //Copio i tensori input in nuovi array per elaborarli
    auto  dimbatch_T_flat = dimbatch_T.flat<int>();
    auto des_buffsize_T_flat = des_buffsize_T.flat<int>();
    auto numpar_T_flat = numpar_T.flat<int>();
    auto num_alpha2b_T_flat = num_alpha2b_T.flat<int>();
    auto nt_T_flat=nt_T.flat<int>();

    int dimbat=dimbatch_T_flat(0);
    int numdes=des_buffsize_T_flat(0);
    int N=numpar_T_flat(0);
    int dimdes=dimbat*numdes*N;
    int nalpha_r=num_alpha2b_T_flat(0);
    int nt=nt_T_flat(0);



    // Create an output tensor
    Tensor* twobodyafs_T = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (dimbat);
    grad_net_shape.AddDim (N);
    grad_net_shape.AddDim (nalpha_r);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &twobodyafs_T));

    //It seems tensorflow does not set to zero the pointed memeory!

    //Calcolo della proiezione su base
    computeSmoothMax(radiale_T,numdes,alpha_radiale_T,nalpha_r,
                     twobodyafs_T,dimbat,N,intmap2b_T,type_emb2b_T,nt,type_map_T);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeSortProj").Device(DEVICE_CPU), ComputeSortProjOp);
