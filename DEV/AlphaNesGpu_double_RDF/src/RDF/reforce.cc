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
    .Output("rdf: double");

void compute_rdf(const double* energy,const double* position,const double* box,int nf, int N,
    const double* ground_energy,const int* type_pair,const int* type, 
    const int* type_map,const double* bin_size,const double* betarewe,double* RDF);
void set_tensor_to_zero_double_cpu(double* vec ,int dimtens){
     for (int g=0;g<dimtens;g++){
         vec[g]=0.;
     }     
}

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
    auto binsize = binsize_T.flat<double>();
    auto type_pair = type_pair_T.flat<int>();
    auto energy = energy_T.flat<double>();
    auto position = position_T.flat<double>();
    auto box = box_T.flat<double>();
    auto ground_energy = ground_energy_T.flat<double>();
    auto betarewe = betarewe_T.flat<double>();
    auto type = type_T.flat<int>();
    auto type_map = type_map_T.flat<int>(); 
    //Prendo le dimensioni del tensore
    int dimbat = position_T.shape().dim_size(0);

    int N = (int)position_T.shape().dim_size(1)/3;


    //Computing number od discrete points 
    const double* boxone;
    double min_box_size=1000000000.;
    const double* box_gl=box.data();
	for (int i=0;i<dimbat;i++)
	{
            boxone=&box_gl[i*6];

		if (boxone[0]<min_box_size)
			min_box_size=boxone[0];
		if (boxone[3]<min_box_size)
			min_box_size=boxone[3];
		if (boxone[5]<min_box_size)
			min_box_size=boxone[5];
    }

    int numdiscr=(int)min_box_size/binsize.data()[0];
    // Create an output tensor
    Tensor* RDF_T = NULL;
    TensorShape RDF_shape ;
    RDF_shape.AddDim (numdiscr);
    OP_REQUIRES_OK(context, context->allocate_output(0, RDF_shape,
                                                     &RDF_T));
    set_tensor_to_zero_double_cpu(RDF_T->flat<double>().data(),numdiscr);

    //Computing three-body atomic-fingerprints
    compute_rdf(energy.data(),position.data(),box.data(),dimbat,N,ground_energy.data(),type_pair.data(),type.data(),type_map.data(),binsize.data(),betarewe.data(),RDF_T->flat<double>().data());
   
}
};
REGISTER_KERNEL_BUILDER(Name("ComputeRdf").Device(DEVICE_CPU), ComputeRdfOp);
