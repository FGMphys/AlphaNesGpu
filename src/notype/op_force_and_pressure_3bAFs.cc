#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>




void compbuff4der(double* desr, double* des3bod,int* intmap_a,double* mu, double* zeta, double* Bp, double* delta1,int N, int b, int par,int nr, int na, double alpha1, double alpha2,double beta1)
{
      //The calculus of descriptor derivatives has been built by buffer mu, zeta, Bp, Cp, delta
      // and here we are computing all of these, before passing them to forces routine
      int nne3bod=intmap_a[b*N*(na*2+1)+par*(na*2+1)];
      int actual=b*N*nr+par*nr;
      int actual_ang=b*N*na+par*na;
      int aux2=0;
      //Initializing buffers with zeros
      mu[0]=0.;
      zeta[0]=0.;
      for (int i=0; i<nr; i++){
          Bp[i]=0.;
      }
      for (int i=0; i<na; i++){
          delta1[i]=0.;
      }
      //Computing whole mu, whole delta, whole Bp
      for (int j=0;j<nne3bod-1;j++){
          for (int y=j+1;y<nne3bod;y++){
              double angulardes=des3bod[actual_ang+aux2];
              double sim1=exp(alpha2*desr[actual+j]+alpha1*desr[actual+y]);
              double sim2=exp(alpha1*desr[actual+j]+alpha2*desr[actual+y]);
              double sum=sim1+sim2;
              double expbeta=exp(beta1*angulardes);
              delta1[aux2]=expbeta*(1.+beta1*angulardes)*sum*0.5;
              double suppj=(alpha1*sim2+alpha2*sim1)*expbeta*0.5;
              double suppy=(alpha1*sim1+alpha2*sim2)*expbeta*0.5;
              Bp[j]+=suppj*angulardes;
              Bp[y]+=suppy*angulardes;
              aux2++;
            }
       }
 }


unsigned int OMPThreadsNum=8;
void setOMPthreads(unsigned int num){OMPThreadsNum=num;};


void computepress_tripl(double* netderiv, double* desr, double* desa, double* intderiv_r,
  double* intderiv_a, int* intmap_r, int* intmap_a, int nr, int na, int N, int dimbat,
  double* force,double* mu, double* zeta, double* Bp, double* delta1,
  int nsmooth_a,double* smooth_a,double* pos,double* box,double* press){
   #pragma omp parallel for num_threads(OMPThreadsNum)
   for (int b=0; b<dimbat; b++){
       for (int par=0; par<N; par++){
           int nne3bod=intmap_a[b*N*(na*2+1)+par*(na*2+1)];
           int na_real=nne3bod*(nne3bod-1)/2;
           int actual=b*N*nr+par*nr;
           int actual_ang=b*N*na+par*na;
           int actgrad=b*N*nsmooth_a+par*nsmooth_a;
           for (int a1=0; a1<nsmooth_a; a1++){
                   double beta=smooth_a[a1*3+2];
                   double alpha1=smooth_a[a1*3];
                   double alpha2=smooth_a[a1*3+1];
                   compbuff4der(desr,desa,intmap_a,mu, zeta, Bp,delta1, N, b, par, nr, na, alpha1, alpha2,beta);
                 for (int nn=0; nn<na_real; nn++){
                     int neighj=intmap_a[b*(N*(na*2+1))+(na*2+1)*par+nn*2+1];
                     int neighk=intmap_a[b*(N*(na*2+1))+(na*2+1)*par+nn*2+2];
                     for (int cor=0; cor<3; cor++){
                        //ottengo il vettore distanza
                        double dist_ij=pos[b*3*N+par*3+cor]-pos[b*3*N+neighj*3+cor];
                        dist_ij-=box[b*4+1+cor]*rint(dist_ij/box[b*4+1+cor]);
                        double dist_ik=pos[b*3*N+par*3+cor]-pos[b*3*N+neighk*3+cor];
                        dist_ik-=box[b*4+1+cor]*rint(dist_ik/box[b*4+1+cor]);

                         double DTlm=0.5*netderiv[actgrad+a1];
                         DTlm*=delta1[nn];
                         double fxij=DTlm*intderiv_a[b*(N*na)*3*2+par*na*3*2+cor*na*2+nn*2];
                         double fxik=DTlm*intderiv_a[b*(N*na)*3*2+par*na*3*2+cor*na*2+nn*2+1];
                         force[b*N*3+par*3+cor]-=(fxij+fxik);
                         force[b*N*3+neighj*3+cor]+=fxij;
                         force[b*N*3+neighk*3+cor]+=fxik;

                         press[b]-=fxij*dist_ij;
                         press[b]-=fxik*dist_ik;
                     }
                 }
                 for (int nn=0; nn<nne3bod;nn++){
                     int neighj=intmap_r[b*(N*(nr+1))+(nr+1)*par+1+nn];
                     double DDp = 0.5*netderiv[actgrad+a1];
		                 DDp*=Bp[nn];
                     for (int cor=0; cor<3; cor++){
                     double dist_ij=pos[b*3*N+par*3+cor]-pos[b*3*N+neighj*3+cor];
                     dist_ij-=box[b*4+1+cor]*rint(dist_ij/box[b*4+1+cor]);
                     double fxi= DDp*intderiv_r[b*N*3*nr+nr*3*par+cor*nr+nn];
                     force[b*(N*3)+3*par+cor] -= fxi;
                     force[b*(N*3)+3*neighj+cor] += fxi;

                     press[b]-=fxi*dist_ij;



                     }
                 }

           }
        }
    }

}


















using namespace tensorflow;

REGISTER_OP("ComputePressTripl")
    .Input("netderiv: double")
    .Input("desr: double")
    .Input("desa: double")
    .Input("intder_r: double")
    .Input("intder_a: double")
    .Input("intmap_r: int32")
    .Input("intmap_a: int32")
    .Input("numdesr: int32")
    .Input("numdesa: int32")
    .Input("numpar: int32")
    .Input("dimbatch: int32")
    .Input("nsmooth_a: int32")
    .Input("smooth_a: double")
    .Input("position: double")
    .Output("force: double")
    .Output("pressure: double");

class ComputePressTriplOp : public OpKernel {
 public:
  explicit ComputePressTriplOp(OpKernelConstruction* context) : OpKernel(context) {}
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
    const Tensor& input_tensor11 = context->input(10);
    const Tensor& input_tensor12 = context->input(11);
    const Tensor& input_tensor13 = context->input(12);
    const Tensor& input_tensor14 = context->input(13);


    //flatting the tensor
    auto input1 = input_tensor1.flat<double>();
    auto input2 = input_tensor2.flat<double>();
    auto input3 = input_tensor3.flat<double>();
    auto input4 = input_tensor4.flat<double>();
    auto input5 = input_tensor5.flat<double>();
    auto input6 = input_tensor6.flat<int>();
    auto input7 = input_tensor7.flat<int>();
    auto input8 = input_tensor8.flat<int>();
    auto input9 = input_tensor9.flat<int>();
    auto input10 = input_tensor10.flat<int>();
    auto input11 = input_tensor11.flat<int>();
    auto input12 = input_tensor12.flat<int>();
    auto input13 = input_tensor13.flat<double>();
    auto input14 = input_tensor14.flat<double>();
    //COPYING ALL TENSOR IN ARRAY TO OPERATE ON THEM

    //dimansion parameters
    int dimbat=input11(0);
    int N=input10(0);
    int nsmooth_a=input12(0);
    int na=input9(0);
    int nr=input8(0);
    int netderivdim=dimbat*N*nsmooth_a;
    //descriptors derivatives and interactions
    //Gradient of NN
    double* net_deriv;
    net_deriv=(double*)malloc(sizeof(double)*netderivdim);
    for (int i = 0; i < netderivdim; i++) {
    net_deriv[i]=input1(i);
    }
    //radial descriptors
    double* radiale;
    radiale=(double*)malloc(sizeof(double)*dimbat*N*nr);
    for (int i = 0; i < dimbat*N*nr; i++) {
    radiale[i]=input2(i);
    }
    //3body descriptors
    double* angolare;
    angolare=(double*)malloc(sizeof(double)*dimbat*N*na);
    for (int i = 0; i < dimbat*N*na; i++) {
    angolare[i]=input3(i);
    }
    //radial derivatives
    double* intderiv_r;
    intderiv_r=(double*)malloc(sizeof(double)*dimbat*N*nr*3);
    for (int i = 0; i < dimbat*N*nr*3; i++) {
    intderiv_r[i]=input4(i);
    }
    //3body derivatives
    double* intderiv_a;
    intderiv_a=(double*)malloc(sizeof(double)*dimbat*N*na*3*2);
    for (int i = 0; i < dimbat*N*na*3*2; i++) {
    intderiv_a[i]=input5(i);
    }
    //2body interactions
    int* intmap_r;
    intmap_r=(int*)malloc(sizeof(int)*dimbat*N*(nr+1));
    for (int i = 0; i < dimbat*N*(nr+1); i++) {
    intmap_r[i]=input6(i);
    }
    //3body interactions
    int* intmap_a;
    intmap_a=(int*)malloc(sizeof(int)*dimbat*N*(na*2+1));
    for (int i = 0; i < dimbat*N*(na*2+1); i++) {
    intmap_a[i]=input7(i);
    }

    double* smooth_a=(double*)malloc(nsmooth_a*3*sizeof(double));
    for (int i=0; i< nsmooth_a*3; i++){
        smooth_a[i]=input13(i);
        }


        //Leggo le posizioni e la scatola
    double* pos;
    pos=(double*)malloc(sizeof(double)*dimbat*3*N);
    double* box;
    box=(double*)malloc(sizeof(double)*dimbat*4);
    for (int i = 0; i < dimbat; i++) {
       for (int elbox=0; elbox<4;elbox++){
              box[i*4+elbox]=input14(i*(3*N+5)+1+elbox);
       }
       for (int par=0; par<3*N; par++){
             pos[i*3*N+par]=input14(i*(3*N+5)+5+par);
       }
    }



    //Allocating buffers and vectors for computing hard derivatives
    //WARNING: Allocation is going to be made, but this kind of allocation is valid only
    //for the training procedure cause we know that number of neighbours never exceeds the
    // the buffer of radial descriptors. Consider other allocation for Molecular Dynamics!
    double* mu=(double*)malloc(sizeof(double));
    double* zeta=(double*)malloc(sizeof(double));
    double* Bp=(double*)malloc(sizeof(double)*nr);
    double* delta1=(double*)malloc(sizeof(double)*na);

    //Allocating the output vector
    double* force;
    force=(double*)malloc(sizeof(double)*dimbat*N*3);
    for (int i=0; i< dimbat*N*3; i++){
        force[i]=0.;
    }
    double* press;
    press=(double*)malloc(sizeof(double)*dimbat);
    for (int i=0; i< dimbat; i++){
        press[i]=0.;
    }
   //COMPUTING FORCES
   computepress_tripl(net_deriv, radiale, angolare, intderiv_r, intderiv_a,
     intmap_r, intmap_a, nr, na, N, dimbat, force, mu, zeta, Bp,
     delta1,nsmooth_a,smooth_a,pos,box,press);




    // Create an output tensor for forces
    Tensor* forces3b = NULL;
    Tensor* press3b = NULL;

    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (dimbat);
    grad_net_shape.AddDim (N);
    grad_net_shape.AddDim (3);
    TensorShape grad_net_press_shape;
    grad_net_press_shape.AddDim (dimbat);

    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &forces3b));
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_net_press_shape,
                                                     &press3b));
    auto forces3bflat = forces3b->flat<double>();
    auto press3bflat = press3b->flat<double>();








    //Copying computed force in Tensorflow framework
    for (int i = 0; i < (dimbat*N*3); i++) {
    forces3bflat(i)=force[i];
    }
      for (int i=0; i<dimbat;i++){
        press3bflat(i)=press[i]/(3*box[i*4+1]*box[i*4+2]*box[i*4+3]);
    }



    //Free memory
    free(pos);
    free(box);
    free(press);
    free(force);
    free(radiale);
    free(angolare);
    free(net_deriv);
    free(intderiv_r);
    free(intderiv_a);
    free(intmap_r);
    free(intmap_a);
    //Free buffer
    free(mu);
    free(zeta);
    free(Bp);
    free(delta1);
    //free alphas
    free(smooth_a);
    }
};
REGISTER_KERNEL_BUILDER(Name("ComputePressTripl").Device(DEVICE_CPU), ComputePressTriplOp);
