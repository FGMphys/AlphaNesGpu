#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"



using namespace tensorflow;

REGISTER_OP("ComputeRdf")
    .Input("energy: double")
    .Input("positions: double")
    .Input("boxes: double")
    .Input("groundenergy: double")
    .Input("typepair: int32")
    .Input("type: int32")
    .Input("typemap: int32")
    .Input("binsize: double")
    .Input("betarewe: double")
    .Output("RDF: double");

void compute_rdf(double* energy,double* position,double* box,double* ground_energy,int* type_pair,int* type, int* type_map,double* bin_size,double* betarewe,double* RDF);
void set_tensor_to_zero_double_cpu(double* vec ,int dimtens);

class ComputeRdfOp : public OpKernel {
 public:
  explicit ComputeRdfOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& energy_T = context->input(0);
    const Tensor& position_T = context->input(1);
    const Tensor& box_T = context->input(2);
    const Tensor& ground_energy_T = context->input(3);
    const Tensor& type_pair_T = context->input(4);
    const Tensor& type_T = context->input(5);
    const Tensor& type_map_T = context->input(6);
    const Tensor& binsize_T = context->input(7);    
    const Tensor& betarewe_T = context->input(8);


    //flattizzo
    auto binsize = binsize_T.lat<double>();
    auto type_pair = type_pair_T.flat<int>();
    auto energy = energy_T.flat<double>();
    auto position = position_T.flat<double>();
    auto box = box_T.flat<double>();
    auto ground_energy = ground_energy_T.flat<double>();
    auto betarewe = betarewe_T.flat<double>();
    auto type = type_T.flat<int>();
    auto type_map = type_map_T.flat<int>(); 
    //Prendo le dimensioni del tensore
    int dimbat = positions.shape().dim_size(0);

    int N = (int)positions.shape().dim_size(2)/3;

    // Create an output tensor
    Tensor* RDF_T = NULL;
    TensorShape RDF_shape ;
    angular_AFs_shape.AddDim (numdiscr);
    OP_REQUIRES_OK(context, context->allocate_output(0, RDF_shape,
                                                     &RDF_T));
    set_tensor_to_zero_double_cpu(RDF_T->flat<double>().data(),dimbat*Nlocal*nsmooth_a);

    //Computing three-body atomic-fingerprints
    compute_rdf(energy.data(),position.data(),box.data(),ground_energy.data(),type_pair.data(),type.data(),type_map.data(),bin_size.data(),betarewe.data(),RDF->flat<double>().data());
   
}
};
REGISTER_KERNEL_BUILDER(Name("ComputeRdf").Device(DEVICE_GPU), ComputeRdfOp);
