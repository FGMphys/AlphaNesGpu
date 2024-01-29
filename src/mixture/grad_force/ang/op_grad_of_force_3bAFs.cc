#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>

using namespace tensorflow;

void computenextgrad_tripl(const Tensor& prevgrad_T, const Tensor& desr_T, const Tensor& desa_T,
  const Tensor& intderiv_r_T, const Tensor& intderiv_a_T, const Tensor& intmap_r_T,
  const Tensor& intmap_a_T, int nr,int na, int N, int dimbat,
  const Tensor& smooth_a_T, int nsmooth_a,int** type_emb3b,const Tensor& type_emb3b_param_T,
  const Tensor& netderiv_T,int nt,const Tensor& type_map_T,const Tensor& tipos_T,
  const Tensor& actual_type_T,Tensor* gradnet_3b_T,Tensor* grad_alpha3b_T,
  Tensor* grad_emb3b_T){


    auto gradnet_3b_T_flat = gradnet_3b_T->flat<double>();
    auto grad_alpha3b_T_flat = grad_alpha3b_T->flat<double>();
    //auto grad_emb3b_T_flat = grad_emb3b_T->shaped<double,2>({nt,nt});
    auto grad_emb3b_T_flat = grad_emb3b_T->flat<double>();

    auto prevgrad_T_flat=prevgrad_T.flat<double>();
    auto netderiv_T_flat=netderiv_T.flat<double>();
    auto desr_T_flat=desr_T.flat<double>();
    auto desa_T_flat=desa_T.flat<double>();
    auto intderiv_r_T_flat=intderiv_r_T.flat<double>();
    auto intderiv_a_T_flat=intderiv_a_T.flat<double>();
    auto intmap_r_T_flat=intmap_r_T.flat<int>();
    auto intmap_a_T_flat=intmap_a_T.flat<int>();
    auto smooth_a_T_flat=smooth_a_T.flat<double>();
    auto type_map_T_flat=type_map_T.flat<int>();
    auto tipos_T_flat=tipos_T.flat<int>();
    auto actual_type_T_flat=actual_type_T.flat<int>();
    auto type_emb3b_param=type_emb3b_param_T.flat<double>();

    int actual_type=actual_type_T_flat(0);
    int nt_couple=nt+nt*(nt-1)/2;

    int tipos_shift=0;
    for (int y=0;y<actual_type;y++){
        tipos_shift=tipos_shift+tipos_T_flat(y);
    }

    int N_local=tipos_T_flat(actual_type);

    for (int i = 0; i < (dimbat*N_local*nsmooth_a); i++) {
    gradnet_3b_T_flat(i)=0.;
    }
    for (int i = 0; i < (nt_couple*nsmooth_a*3); i++) {
    grad_alpha3b_T_flat(i)=0.;
    }
    for (int i = 0; i < nt_couple*nsmooth_a; i++) {
        grad_emb3b_T_flat(i)=0.;

    }


   for (int b=0; b<dimbat; b++){
       for (int par=0; par<N_local; par++){
          int absolute_par=par+tipos_shift;
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
                   int j_type=type_map_T_flat(neighj);
                   int k_type=type_map_T_flat(neighk);
                   int cht_where=type_emb3b[j_type][k_type];
                   int sum=cht_where;


                   double angulardes=desa_T_flat(actual_ang+nn);
                   double radialdes_j=desr_T_flat(actual+j);
                   double radialdes_k=desr_T_flat(actual+k);


                    for (int cor=0; cor<3; cor++){
                      double intder_j=intderiv_a_T_flat(b*(N_local*na)*3*2+par*na*3*2+cor*na*2+nn*2);
                      double intder_k=intderiv_a_T_flat(b*(N_local*na)*3*2+par*na*3*2+cor*na*2+nn*2+1);

                      double intder_r_j=intderiv_r_T_flat(b*N_local*3*nr+nr*3*par+cor*nr+j);
                      double intder_r_k=intderiv_r_T_flat(b*N_local*3*nr+nr*3*par+cor*nr+k);


                       double prevgrad_loc=prevgrad_T_flat(b*(N*3)+absolute_par*3+cor);
                       double prevgrad_neighj=prevgrad_T_flat(b*(N*3)+neighj*3+cor);
                       double prevgrad_neighk=prevgrad_T_flat(b*(N*3)+neighk*3+cor);


                       for (int a1=0; a1<nsmooth_a; a1++){
                           double alpha1=smooth_a_T_flat(sum*nsmooth_a*3+a1*3+0);
                           double alpha2=smooth_a_T_flat(sum*nsmooth_a*3+a1*3+1);
                           double beta=smooth_a_T_flat(sum*nsmooth_a*3+a1*3+2);
                           double chtjk_par=type_emb3b_param(cht_where*nsmooth_a+a1);

                           double expbeta=exp(beta*angulardes);

                           double sim1=exp(alpha2*radialdes_j+alpha1*radialdes_k);
                           double sim2=exp(alpha1*radialdes_j+alpha2*radialdes_k);
                           double sum_sim=sim1+sim2;

                           double delta=expbeta*(1.+beta*angulardes)*sum_sim*0.5;

                           double suppj=(alpha1*sim2+alpha2*sim1)*expbeta;
                           double suppk=(alpha1*sim1+alpha2*sim2)*expbeta;
                           double Bp_j=suppj*angulardes*0.5;
                           double Bp_k=suppk*angulardes*0.5;

                           double gradxij=chtjk_par*delta*intder_j+chtjk_par*Bp_j*intder_r_j;
                           double gradxik=chtjk_par*delta*intder_k+chtjk_par*Bp_k*intder_r_k;
                           if (b==0 && par==0 && cor==0 && nn==0 && actual_type==0)
			      printf("prevgrad_loc %g gradxij %g gradxik %g at b %d par %d cor %d nn %d a1 %d\n",prevgrad_loc,gradxij,gradxik,b,par,cor,nn,a1);
                           gradnet_3b_T_flat(actgrad+a1)-=prevgrad_loc*0.5*(gradxij+gradxik);
                           gradnet_3b_T_flat(actgrad+a1)+=prevgrad_neighj*0.5*gradxij;
                           gradnet_3b_T_flat(actgrad+a1)+=prevgrad_neighk*0.5*gradxik;

                           double grad_emb_xij=(delta*intder_j+Bp_j*intder_r_j);
                           double grad_emb_xik=(delta*intder_k+Bp_k*intder_r_k);

                           double NGel=netderiv_T_flat(actgrad+a1);



                           grad_emb3b_T_flat(cht_where*nsmooth_a+a1)-=prevgrad_loc*0.5*NGel*(grad_emb_xij+grad_emb_xik);
                           grad_emb3b_T_flat(cht_where*nsmooth_a+a1)+=prevgrad_neighj*0.5*NGel*grad_emb_xij;
                           grad_emb3b_T_flat(cht_where*nsmooth_a+a1)+=prevgrad_neighk*0.5*NGel*grad_emb_xik;



                           double buff_a1_ang=expbeta*(1.+beta*angulardes)*(sim1*radialdes_k+sim2*radialdes_j)*0.5;
                           double buff_a2_ang=expbeta*(1.+beta*angulardes)*(sim1*radialdes_j+sim2*radialdes_k)*0.5;
                           double buff_beta_ang=expbeta*angulardes*(2.+beta*angulardes)*sum_sim*0.5;

                           double buff_beta_r_j=suppj*angulardes*angulardes*0.5;
                           double buff_beta_r_k=suppk*angulardes*angulardes*0.5;

	                   double buff_a1_r_j=(sim2+alpha1*sim2*radialdes_j+alpha2*sim1*radialdes_k)*expbeta*0.5*angulardes;
                           double buff_a2_r_j=(sim1+alpha2*sim1*radialdes_j+alpha1*sim2*radialdes_k)*expbeta*0.5*angulardes;

                           double buff_a1_r_k=(sim1+alpha1*sim1*radialdes_k+alpha2*sim2*radialdes_j)*expbeta*0.5*angulardes;
                           double buff_a2_r_k=(sim2+alpha2*sim2*radialdes_k+alpha1*sim1*radialdes_j)*expbeta*0.5*angulardes;

                           double grad_a1_xij=chtjk_par*buff_a1_ang*intder_j+chtjk_par*buff_a1_r_j*intder_r_j;
                           double grad_a1_xik=chtjk_par*buff_a1_ang*intder_k+chtjk_par*buff_a1_r_k*intder_r_k;

                           double grad_a2_xij=chtjk_par*buff_a2_ang*intder_j+chtjk_par*buff_a2_r_j*intder_r_j;
                           double grad_a2_xik=chtjk_par*buff_a2_ang*intder_k+chtjk_par*buff_a2_r_k*intder_r_k;

                           double grad_beta_xij=chtjk_par*buff_beta_ang*intder_j+chtjk_par*buff_beta_r_j*intder_r_j;
                           double grad_beta_xik=chtjk_par*buff_beta_ang*intder_k+chtjk_par*buff_beta_r_k*intder_r_k;




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
    .Input("type_emb3b_parameters: double")
    .Input("number_of_atom_type: int32")
    .Input("type_map: int32")
    .Input("tipos: int32")
    .Input("actual_type: int32")
    .Output("nextgrad_ng: double")
    .Output("nextgrad_alpha: double")
    .Output("next_emb3b_grad: double");
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

    const Tensor& type_emb3b_parameters_T = context->input(14);
    const Tensor& number_of_atom_type_T = context->input(15);
    const Tensor& type_map_T = context->input(16);

    const Tensor& tipos_T = context->input(17);
    const Tensor& actual_type_T = context->input(18);


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


    auto nt_flat = number_of_atom_type_T.flat<int>();
    int nt=nt_flat(0);
    int nt_couple=nt+nt*(nt-1)/2;

    //Deal with symmetric parameter matrix of type embedding 3body
    //generate symmetric type embedding
   int** type_emb3b=(int**)calloc(nt,sizeof(int*));
   for (int k=0;k<nt;k++){
       type_emb3b[k]=(int*)calloc(nt,sizeof(int));
      }
   int k=0;
   for (int i = 0; i < nt; i++) {
       for (int j = i; j < nt; j++){
           type_emb3b[i][j]=k;//type_emb3b_parameters_T_flat(k);
           type_emb3b[j][i]=k;//type_emb3b_parameters_T_flat(k);
           k=k+1;
         }
   }


    auto tipos_T_flat=tipos_T.flat<int>();
    auto actual_type_T_flat=actual_type_T.flat<int>();
    int actual_type=actual_type_T_flat(0);
    int N_local=tipos_T_flat(actual_type);

    double** nextgrad_emb3b_sym=(double**)calloc(nt,sizeof(double*));
    for (int k=0;k<nt;k++){
        nextgrad_emb3b_sym[k]=(double*)calloc(nt,sizeof(double));
       }

    // Create an output tensor for gradient wrt to NG
    Tensor* gradnet_3b_T = NULL;
    TensorShape gradnet_3b_shape ;
    gradnet_3b_shape.AddDim (1);
    gradnet_3b_shape.AddDim (dimbat);
    gradnet_3b_shape.AddDim (N_local);
    gradnet_3b_shape.AddDim (nsmooth_a);
    OP_REQUIRES_OK(context, context->allocate_output(0, gradnet_3b_shape,
                                                     &gradnet_3b_T));

    // Create an output tensor for gradient wrt to alphas
    Tensor* grad_alpha3b_T = NULL;
    TensorShape grad_alpha3b_shape ;
    grad_alpha3b_shape.AddDim (nt_couple);
    grad_alpha3b_shape.AddDim (3*nsmooth_a);
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_alpha3b_shape,
                                                     &grad_alpha3b_T));


    // Create an output tensor for gradient wrt to alphas
    Tensor* grad_emb3b_T = NULL;
    TensorShape grad_emb3b_shape;
    grad_emb3b_shape.AddDim (nt_couple);
    grad_emb3b_shape.AddDim (nsmooth_a);

    OP_REQUIRES_OK(context, context->allocate_output(2, grad_emb3b_shape,
                                                     &grad_emb3b_T));





    //COMPUTING NEXTGRAD
   computenextgrad_tripl(prevgrad_T,radial_descriptor_T,angular_descriptor_T,
                          desder_rad_T, desder_ang_T,interaction_map_rad_T,
                          interaction_map_angular_T,nr, na, N, dimbat,
                          alpha3b_parameters_T,nsmooth_a,type_emb3b,
                          type_emb3b_parameters_T,netderiv_T,nt,
                          type_map_T,tipos_T,actual_type_T,gradnet_3b_T,
                          grad_alpha3b_T,grad_emb3b_T);



    for (int k=0;k<nt;k++){
         free(type_emb3b[k]);
    }
    free(type_emb3b);


    }
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceTriplGrad").Device(DEVICE_CPU), ComputeForceTriplGradOp);
