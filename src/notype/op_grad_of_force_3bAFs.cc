#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>

using namespace tensorflow;

void computenextgrad_tripl(const Tensor& prevgrad_T, const Tensor& desr_T, const Tensor& desa_T,
  const Tensor& intderiv_r_T, const Tensor& intderiv_a_T, const Tensor& intmap_r_T,
  const Tensor& intmap_a_T, int nr,int na, int N, int dimbat,
  const Tensor& smooth_a_T, int nsmooth_a,const Tensor& netderiv_T,
  Tensor* gradnet_3b_T,Tensor* grad_alpha3b_T){


    auto gradnet_3b_T_flat = gradnet_3b_T->flat<double>();
    auto grad_alpha3b_T_flat = grad_alpha3b_T->flat<double>();

    auto prevgrad_T_flat=prevgrad_T.flat<double>();
    auto netderiv_T_flat=netderiv_T.flat<double>();
    auto desr_T_flat=desr_T.flat<double>();
    auto desa_T_flat=desa_T.flat<double>();
    auto intderiv_r_T_flat=intderiv_r_T.flat<double>();
    auto intderiv_a_T_flat=intderiv_a_T.flat<double>();
    auto intmap_r_T_flat=intmap_r_T.flat<int>();
    auto intmap_a_T_flat=intmap_a_T.flat<int>();
    auto smooth_a_T_flat=smooth_a_T.flat<double>();




    int N_local=N;

    for (int i = 0; i < (dimbat*N_local*nsmooth_a); i++) {
    gradnet_3b_T_flat(i)=0.;
    }
    for (int i = 0; i < (nsmooth_a*3); i++) {
    grad_alpha3b_T_flat(i)=0.;
    }


   for (int b=0; b<dimbat; b++){
       for (int par=0; par<N_local; par++){
          int nne3bod=intmap_a_T_flat(b*N_local*(na*2+1)+par*(na*2+1));
          int na_real=nne3bod*(nne3bod-1)/2;
          int actual=b*N_local*nr+par*nr;
          int actual_ang=b*N_local*na+par*na;
          int actgrad=b*N_local*nsmooth_a+par*nsmooth_a;
          int nn=0;

           for (int j=0;j<nne3bod-1;j++){
               for (int k=j+1;k<nne3bod;k++){
                   int neighj=intmap_a_T_flat(b*(N_local*(na*2+1))+(na*2+1)*par+nn*2+1);
                   int neighk=intmap_a_T_flat(b*(N_local*(na*2+1))+(na*2+1)*par+nn*2+2);

                   int sum=0;

                   double angulardes=desa_T_flat(actual_ang+nn);
                   double radialdes_j=desr_T_flat(actual+j);
                   double radialdes_k=desr_T_flat(actual+k);


                    for (int cor=0; cor<3; cor++){
                      double intder_j=intderiv_a_T_flat(b*(N_local*na)*3*2+par*na*3*2+cor*na*2+nn*2);
                      double intder_k=intderiv_a_T_flat(b*(N_local*na)*3*2+par*na*3*2+cor*na*2+nn*2+1);

                      double intder_r_j=intderiv_r_T_flat(b*N_local*3*nr+nr*3*par+cor*nr+j);
                      double intder_r_k=intderiv_r_T_flat(b*N_local*3*nr+nr*3*par+cor*nr+k);


                       double prevgrad_loc=prevgrad_T_flat(b*(N*3)+par*3+cor);
                       double prevgrad_neighj=prevgrad_T_flat(b*(N*3)+neighj*3+cor);
                       double prevgrad_neighk=prevgrad_T_flat(b*(N*3)+neighk*3+cor);


                       for (int a1=0; a1<nsmooth_a; a1++){
                           double alpha1=smooth_a_T_flat(sum*nsmooth_a*3+a1*3+0);
                           double alpha2=smooth_a_T_flat(sum*nsmooth_a*3+a1*3+1);
                           double beta=smooth_a_T_flat(sum*nsmooth_a*3+a1*3+2);

                           double expbeta=exp(beta*angulardes);

                           double sim1=exp(alpha2*radialdes_j+alpha1*radialdes_k);
                           double sim2=exp(alpha1*radialdes_j+alpha2*radialdes_k);
                           double sum_sim=sim1+sim2;

                           double delta=expbeta*(1.+beta*angulardes)*sum_sim*0.5;

                           double suppj=(alpha1*sim2+alpha2*sim1)*expbeta;
                           double suppk=(alpha1*sim1+alpha2*sim2)*expbeta;
                           double Bp_j=suppj*angulardes*0.5;
                           double Bp_k=suppk*angulardes*0.5;

                           double gradxij=delta*intder_j+Bp_j*intder_r_j;
                           double gradxik=delta*intder_k+Bp_k*intder_r_k;

                           gradnet_3b_T_flat(actgrad+a1)-=prevgrad_loc*0.5*(gradxij+gradxik);
                           gradnet_3b_T_flat(actgrad+a1)+=prevgrad_neighj*0.5*gradxij;
                           gradnet_3b_T_flat(actgrad+a1)+=prevgrad_neighk*0.5*gradxik;


                           double NGel=netderiv_T_flat(actgrad+a1);

                           double buff_a1_ang=expbeta*(1.+beta*angulardes)*(sim1*radialdes_k+sim2*radialdes_j)*0.5;
                           double buff_a2_ang=expbeta*(1.+beta*angulardes)*(sim1*radialdes_j+sim2*radialdes_k)*0.5;
                           double buff_beta_ang=expbeta*angulardes*(2.+beta*angulardes)*sum_sim*0.5;

                           double buff_beta_r_j=suppj*angulardes*angulardes*0.5;
                           double buff_beta_r_k=suppk*angulardes*angulardes*0.5;

			                     double buff_a1_r_j=(sim2+alpha1*sim2*radialdes_j+alpha2*sim1*radialdes_k)*expbeta*0.5*angulardes;
                           double buff_a2_r_j=(sim1+alpha2*sim1*radialdes_j+alpha1*sim2*radialdes_k)*expbeta*0.5*angulardes;

                           double buff_a1_r_k=(sim1+alpha1*sim1*radialdes_k+alpha2*sim2*radialdes_j)*expbeta*0.5*angulardes;
                           double buff_a2_r_k=(sim2+alpha2*sim2*radialdes_k+alpha1*sim1*radialdes_j)*expbeta*0.5*angulardes;

                           double grad_a1_xij=buff_a1_ang*intder_j+buff_a1_r_j*intder_r_j;
                           double grad_a1_xik=buff_a1_ang*intder_k+buff_a1_r_k*intder_r_k;

                           double grad_a2_xij=buff_a2_ang*intder_j+buff_a2_r_j*intder_r_j;
                           double grad_a2_xik=buff_a2_ang*intder_k+buff_a2_r_k*intder_r_k;

                           double grad_beta_xij=buff_beta_ang*intder_j+buff_beta_r_j*intder_r_j;
                           double grad_beta_xik=buff_beta_ang*intder_k+buff_beta_r_k*intder_r_k;




                           grad_alpha3b_T_flat(sum*nsmooth_a*3+a1*3+0)-=prevgrad_loc*0.5*NGel*(grad_a1_xij+grad_a1_xik);
                           grad_alpha3b_T_flat(sum*nsmooth_a*3+a1*3+0)+=prevgrad_neighj*0.5*NGel*grad_a1_xij;
                           grad_alpha3b_T_flat(sum*nsmooth_a*3+a1*3+0)+=prevgrad_neighk*0.5*NGel*grad_a1_xik;

                           grad_alpha3b_T_flat(sum*nsmooth_a*3+a1*3+1)-=prevgrad_loc*0.5*NGel*(grad_a2_xij+grad_a2_xik);
                           grad_alpha3b_T_flat(sum*nsmooth_a*3+a1*3+1)+=prevgrad_neighj*0.5*NGel*grad_a2_xij;
                           grad_alpha3b_T_flat(sum*nsmooth_a*3+a1*3+1)+=prevgrad_neighk*0.5*NGel*grad_a2_xik;

                           grad_alpha3b_T_flat(sum*nsmooth_a*3+a1*3+2)-=prevgrad_loc*0.5*NGel*(grad_beta_xij+grad_beta_xik);
                           grad_alpha3b_T_flat(sum*nsmooth_a*3+a1*3+2)+=prevgrad_neighj*0.5*NGel*grad_beta_xij;
                           grad_alpha3b_T_flat(sum*nsmooth_a*3+a1*3+2)+=prevgrad_neighk*0.5*NGel*grad_beta_xik;


                       }
                   }
                   nn=nn+1;

        }
      }
    }
  }

}

















REGISTER_OP("ComputeForceTriplGrad")
    .Input("prevgrad: double")
    .Input("netderiv: double")
    .Input("radial_descriptor: double")
    .Input("angular_descriptor: double")
    .Input("descriptor_derivative_rad: double")
    .Input("descriptor_derivative_ang: double")
    .Input("interaction_map_rad: int32")
    .Input("interaction_map_ang: int32")
    .Input("radial_descriptor_buffsize: int32")
    .Input("angular_descriptor_buffsize: int32")
    .Input("number_of_particles_in_frame: int32")
    .Input("dimbatch: int32")
    .Input("number_alpha3b: int32")
    .Input("alpha3b_parameters: double")
    .Output("nextgrad_ng: double")
    .Output("nextgrad_alpha: double");
class ComputeForceTriplGradOp : public OpKernel {
 public:
  explicit ComputeForceTriplGradOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    // Grab the input tensor
    const Tensor& prevgrad_T=context->input(0);
    const Tensor& netderiv_T = context->input(1);
    const Tensor& radial_descriptor_T = context->input(2);
    const Tensor& angular_descriptor_T = context->input(3);
    const Tensor& desder_rad_T = context->input(4);
    const Tensor& desder_ang_T = context->input(5);
    const Tensor& interaction_map_rad_T = context->input(6);
    const Tensor& interaction_map_angular_T = context->input(7);
    const Tensor& radial_descriptor_buffsize_T = context->input(8);
    const Tensor& angular_descriptor_buffsize_T = context->input(9);
    const Tensor& numpar_T = context->input(10);
    const Tensor& batch_dimension_T = context->input(11);
    const Tensor& number_alpha3b_T = context->input(12);
    const Tensor& alpha3b_parameters_T = context->input(13);



    //flatting the tensor
    auto dimbat_flat = batch_dimension_T.flat<int>();
    int dimbat=dimbat_flat(0);

    auto numdes2body_flat = radial_descriptor_buffsize_T.flat<int>();
    int nr=numdes2body_flat(0);

    auto numdes3body_flat = angular_descriptor_buffsize_T.flat<int>();
    int na=numdes3body_flat(0);

    auto N_flat = numpar_T.flat<int>();
    int N=N_flat(0);

    auto nsmooth_a_flat=number_alpha3b_T.flat<int>();
    int nsmooth_a=nsmooth_a_flat(0);



    // Create an output tensor for gradient wrt to NG
    Tensor* gradnet_3b_T = NULL;
    TensorShape gradnet_3b_shape ;
    gradnet_3b_shape.AddDim (1);
    gradnet_3b_shape.AddDim (dimbat);
    gradnet_3b_shape.AddDim (N);
    gradnet_3b_shape.AddDim (nsmooth_a);
    OP_REQUIRES_OK(context, context->allocate_output(0, gradnet_3b_shape,
                                                     &gradnet_3b_T));

    // Create an output tensor for gradient wrt to alphas
    Tensor* grad_alpha3b_T = NULL;
    TensorShape grad_alpha3b_shape ;
    grad_alpha3b_shape.AddDim (nsmooth_a);
    grad_alpha3b_shape.AddDim (3);
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_alpha3b_shape,
                                                     &grad_alpha3b_T));




    //COMPUTING NEXTGRAD
   computenextgrad_tripl(prevgrad_T,radial_descriptor_T,angular_descriptor_T,
                          desder_rad_T, desder_ang_T,interaction_map_rad_T,
                          interaction_map_angular_T,nr, na, N, dimbat,
                          alpha3b_parameters_T,nsmooth_a,netderiv_T,
                          gradnet_3b_T,grad_alpha3b_T);




    }
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceTriplGrad").Device(DEVICE_CPU), ComputeForceTriplGradOp);
