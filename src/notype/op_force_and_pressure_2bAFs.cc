#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"


void computeSmoothMaxForce_Virial(double *ds,int num_ds,double *alpha,int num_alpha_radiale,double *buffer1,
  double *buffer2,double *sds,double **sds_deriv,double* grad, double* intderiv,
  int* nnindex, int dimbat, int N, double* force, double* pos, double* box, double* press)
{
  int par2,b,i;
  int numdes=num_ds;
  // costruiamo i descrittori
  for (int i = 0; i < dimbat*N*3; i++) {
      force[i] = 0.;
    }
  for (int i = 0; i<dimbat; i++){
      press[i]=0.;
  }
for (b=0;b<dimbat;b++){

    for (par2=0;par2<N;par2++){
        int actual=b*N*numdes+par2*numdes;
        int num_neigh=nnindex[b*(N*(numdes+1))+(numdes+1)*par2];
        //Proiezione su base e derivazione del singolo termine proiettato per la par2ticella par2 nel frame b
        for (i=0;i<num_alpha_radiale;i++)
            {
              buffer1[i]=0.;
              buffer2[i]=0.;
              int j;
              for (j=0;j<num_neigh;j++)
              {
                //buffer1[i]+=ds[actual+j]*exp(alpha[i]*ds[actual+j]);
                //buffer2[i]+=exp(alpha[i]*ds[actual+j]);
              }
              //buffer2[i]+=double(N-num_neigh-1);
              //sds[i]=buffer1[i];///buffer2[i];

              for (j=0;j<num_neigh;j++)
              {
                sds_deriv[i][j]=exp(alpha[i]*ds[actual+j])*(1.+alpha[i]*ds[actual+j]);
              }
        }
        //Calcolo del contributo self e non self della forza relativo alla par2ticella par2 nel frame b
        for (int p =0; p<num_neigh; p++){
            int neighj=nnindex[b*(N*(numdes+1))+(numdes+1)*par2+1+p];
        for (int a =0; a<3; a++){
        		//ottengo il vettore distanza
            double dist_a=pos[b*3*N+par2*3+a]-pos[b*3*N+neighj*3+a];
            dist_a-=box[b*4+1+a]*rint(dist_a/box[b*4+1+a]);
            double int_deriv= intderiv[b*N*3*numdes+numdes*3*par2+a*numdes+p];

        for (int alpha=0; alpha<num_alpha_radiale;alpha++){
            double sdsderiv=sds_deriv[alpha][p];
            double gradel=grad[b*N*num_alpha_radiale+num_alpha_radiale*par2+alpha];
            double temp = 0.5*sdsderiv*int_deriv;

            force[b*(N*3)+3*par2+a] -= gradel*temp;
            press[b]-= gradel*temp*dist_a;

            force[b*(N*3)+3*neighj+a] += gradel*temp;

                   }
             }

          }
}
}
}

using namespace tensorflow;

REGISTER_OP("ComputePressRadial")
    .Input("netderiv: double")
    .Input("intderiv: double")
    .Input("nnindex: int32")
    .Input("numpar2: int32")
    .Input("numnn: int32")
    .Input("dimbatch: int32")
    .Input("sortdes: double")
    .Input("nalpha_r: int32")
    .Input("alpharad: double")
    .Input("position: double")
    .Output("force: double")
    .Output("pressure: double");

class ComputePressRadialOp : public OpKernel {
 public:
  explicit ComputePressRadialOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor1 = context->input(0);
    const Tensor& input_tensor2 = context->input(1);
    const Tensor& input_tensor3 = context->input(2);
    const Tensor& input_tensor4 = context->input(3);
    const Tensor& input_tensor5 = context->input(4);
    const Tensor& input_tensor6 = context->input(5);
    const Tensor& input_tensor7 = context->input(6);
    const Tensor& input_tensor8 = context->input(7);
    const Tensor& input_tensor9 = context->input(8);
    const Tensor& input_tensor10 = context->input(9);


    auto input = input_tensor1.flat<double>();
    auto input2 = input_tensor2.flat<double>();
    auto input3 = input_tensor3.flat<int>();
    auto input4 = input_tensor4.flat<int>();
    auto input5 = input_tensor5.flat<int>();
    auto input6 = input_tensor6.flat<int>();
    auto input7 = input_tensor7.flat<double>();
    auto input8 = input_tensor8.flat<int>();
    auto input9 = input_tensor9.flat<double>();
    auto input10 = input_tensor10.flat<double>();

    //Copio i tensori input in nuovi array per elaborarli
    int dimbat=input6(0);
    int numdes=input5(0);
    int N=input4(0);
    int dimdes=dimbat*numdes*N;
    //int max_input=input9(0);
    int num_alpha_radiale=input8(0);
    int dimnetderiv=num_alpha_radiale*dimbat*N;
    //Leggo la derivata di rete
    double* net_deriv;
    net_deriv=(double*)malloc(sizeof(double)*dimnetderiv);
    for (int i = 0; i < dimnetderiv; i++) {
    net_deriv[i]=input(i);
    }
    //Leggo il descrittore di rete
    double* radiale;
    radiale=(double*)malloc(sizeof(double)*dimdes);
    for (int i = 0; i < dimdes; i++) {
    radiale[i]=input7(i);
    }
    //Leggo la derivata del descrittore
    double* int_deriv;
    int_deriv=(double*)malloc(sizeof(double)*dimdes*3);
    for (int i = 0; i < (dimdes*3); i++) {
    int_deriv[i]=input2(i);
    }
    //Leggo la mappa di interazione
    int* nnindex;
    nnindex=(int*)malloc(sizeof(int)*dimbat*(numdes+1)*N);
    for (int i = 0; i < dimbat*(numdes+1)*N; i++) {
    nnindex[i]=input3(i);
    }




    //Leggo le posizioni e la scatola
    double* pos;
    pos=(double*)malloc(sizeof(double)*dimbat*3*N);
    double* box;
    box=(double*)malloc(sizeof(double)*dimbat*4);
    for (int i = 0; i < dimbat; i++) {
       for (int elbox=0; elbox<4;elbox++){
              box[i*4+elbox]=input10(i*(3*N+5)+1+elbox);
       }
       for (int par=0; par<3*N; par++){
             pos[i*3*N+par]=input10(i*(3*N+5)+5+par);
       }
    }




    //Alloco gli array di output
    double* force;
    force=(double*)malloc(sizeof(double)*dimbat*N*3);
    double* press;
    press=(double*)malloc(sizeof(double)*dimbat);
    //Definisco i termini dell'espansione da considerare
    //double alpha_radiale_modmax=input8(0);
    double* alpha_radiale=(double*)malloc(num_alpha_radiale*sizeof(double));
    for (int i=0; i< num_alpha_radiale; i++){
        alpha_radiale[i]=input9(i);
        }
//    buildUniformAlpha(alpha_radiale,num_alpha_radiale,alpha_radiale_modmax);
    double* smoothradiale=(double*)malloc(num_alpha_radiale*sizeof(double));
    double** smoothradialederiv=(double**)malloc(num_alpha_radiale*sizeof(double*));
    for (int k=0;k<num_alpha_radiale;k++){
        smoothradialederiv[k]=(double*)malloc(numdes*sizeof(double));
       }
    double* buffer1=(double*)malloc(num_alpha_radiale*sizeof(double));
    double* buffer2=(double*)malloc(num_alpha_radiale*sizeof(double));
    //Calcolo delle forze
    computeSmoothMaxForce_Virial(radiale,numdes,alpha_radiale,num_alpha_radiale,
      buffer1,buffer2,smoothradiale,smoothradialederiv,net_deriv,int_deriv,
      nnindex,dimbat,N,force,pos,box,press);



    //Calcolo dei descrittori
    //computeforce_r(net_deriv,int_deriv,nnindex,dimbat,N,numdes,force);

    // Create an output tensor
    Tensor* forces2b = NULL;
    Tensor* press2b = NULL;

    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (dimbat);
    grad_net_shape.AddDim (N);
    grad_net_shape.AddDim (3);
    TensorShape grad_net_press_shape;
    grad_net_press_shape.AddDim (dimbat);


    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &forces2b));
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_net_press_shape,
                                                     &press2b));
    auto forces2bflat = forces2b->flat<double>();
    auto press2bflat = press2b->flat<double>();






    //Copio element per elemento l'array di output nel tensore di output
    for (int i = 0; i < (dimbat*N*3); i++) {
    forces2bflat(i)=force[i];
    }
    for (int i=0; i<dimbat;i++){
        press2bflat(i)=press[i]/(3*box[i*4+1]*box[i*4+2]*box[i*4+3]);
    }

    free(force);
    free(pos);
    free(box);
    free(press);
    free(net_deriv);
    free(int_deriv);
    free(nnindex);
    free(buffer1);
    free(buffer2);
    free(smoothradiale);
    for (int k=0;k<num_alpha_radiale;k++){
        free(smoothradialederiv[k]);
       }
    free(smoothradialederiv);
    free(alpha_radiale);
    free(radiale);
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputePressRadial").Device(DEVICE_CPU), ComputePressRadialOp);
