//#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"

#include "vector.h"

#define SQR(x) ((x)*(x))


#define BLOCK_DIM 256

#define PI 3.141592654f

__global__ void DescriptorsRadial_kernel(double range,int radial_buffer,double range_angolare,int angular_buffer,int N,
                      double* position,const double* boxes,
                      int *howmany,int *with,
                      double* descriptors,int* intmap2b,double* der2b,
                      double* des3bsupp,
                      double* der3bsupp, int nf,int* numtriplet,
		                  double rs, double coeffa_intra,double coeffb_intra,double coeffc_intra,double coeffa_inter,double coeffb_inter
                      ,double coeffc_inter,double pow_alpha, double pow_beta,
          double Rs_inter, double Rc_inter, const int* map_intra_d,double Ra_inter)
{
  int t=blockIdx.x*blockDim.x+threadIdx.x;

  // from t to b,par,j,k
  int b=t/(radial_buffer*N);
  int reminder=t%(radial_buffer*N);
  int i=reminder/radial_buffer;
  int k=reminder%radial_buffer;

  double3* coor=(double3*)position;
  // shared memory counter for numero di vicini angolari
  __shared__ int3 num_angolare[BLOCK_DIM];

  num_angolare[threadIdx.x].x=b;
  num_angolare[threadIdx.x].y=i;
  num_angolare[threadIdx.x].z=0;

  if (t<nf*N*radial_buffer)
  {
    if (k==0)
    {
      intmap2b[b*N*(radial_buffer+1)+i*(radial_buffer+1)]=howmany[b*N+i];
    }

    vector olddist,dist;
    double dist_norm;

    if (k<howmany[b*N+i])
    {
      int j=with[b*N*radial_buffer+i*radial_buffer+k];
      //coor is in internal coordinate
      olddist.x=coor[b*N+i].x-coor[b*N+j].x;
      olddist.y=coor[b*N+i].y-coor[b*N+j].y;
      olddist.z=coor[b*N+i].z-coor[b*N+j].z;

      olddist.x-=rint(olddist.x);
      olddist.y-=rint(olddist.y);
      olddist.z-=rint(olddist.z);
      // vectorial distance is bring back in cartesian coordinate
      dist.x=boxes[b*6+0]*olddist.x+boxes[b*6+1]*olddist.y+boxes[b*6+2]*olddist.z;
      dist.y=boxes[b*6+3]*olddist.y+boxes[b*6+4]*olddist.z;
      dist.z=boxes[b*6+5]*olddist.z;

      dist_norm=sqrt(SQR(dist.x)+SQR(dist.y)+SQR(dist.z));

      int actual_pos=b*N*radial_buffer+i*radial_buffer;

      int same=map_intra_d[i]-map_intra_d[j];
      //STEP 0: filling the radial part with the smaller angular cut-off
      if (same==0){
        if (dist_norm<range_angolare)
        {
          num_angolare[threadIdx.x].z=1;

          des3bsupp[actual_pos+k]=0.5*(cos(PI*dist_norm/range_angolare)+1);
          der3bsupp[b*N*3*radial_buffer+i*3*radial_buffer+k]=-0.5*sin(PI*dist_norm/range_angolare)*PI/range_angolare*dist.x/dist_norm;
          der3bsupp[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k]=-0.5*sin(PI*dist_norm/range_angolare)*PI/range_angolare*dist.y/dist_norm;
          der3bsupp[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k]=-0.5*sin(PI*dist_norm/range_angolare)*PI/range_angolare*dist.z/dist_norm;
        }
        //STEP 1: filling radial descriptor with larger cutoff
        if (dist_norm<rs){
            descriptors[actual_pos+k]=coeffa_intra/pow(dist_norm/rs,pow_alpha)+coeffb_intra/pow(dist_norm/rs,pow_beta)+coeffc_intra;

            double der_cutoff=(-pow_alpha*coeffa_intra/pow(dist_norm/rs,pow_alpha+1.)/rs-pow_beta*coeffb_intra/pow(dist_norm/rs,pow_beta+1.)/rs);

            der2b[b*N*3*radial_buffer+i*3*radial_buffer+k]=der_cutoff*dist.x/dist_norm;
            der2b[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k]=der_cutoff*dist.y/dist_norm;
            der2b[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k]=der_cutoff*dist.z/dist_norm;
        }
        else{
          descriptors[actual_pos+k]=0.5*(cos(PI*dist_norm/range)+1);
          der2b[b*N*3*radial_buffer+i*3*radial_buffer+k]=-0.5*sin(PI*dist_norm/range)*PI/range*dist.x/dist_norm;
          der2b[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k]=-0.5*sin(PI*dist_norm/range)*PI/range*dist.y/dist_norm;
          der2b[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k]=-0.5*sin(PI*dist_norm/range)*PI/range*dist.z/dist_norm;
        }
        //STEP 2: filling interaction map for pair descriptors
        intmap2b[b*N*(radial_buffer+1)+i*(radial_buffer+1)+1+k]=with[b*radial_buffer*N+i*radial_buffer+k];
    }
    else {
      if (dist_norm<Ra_inter)
      {
        num_angolare[threadIdx.x].z=1;

        des3bsupp[actual_pos+k]=0.5*(cos(PI*dist_norm/Ra_inter)+1);
        der3bsupp[b*N*3*radial_buffer+i*3*radial_buffer+k]=-0.5*sin(PI*dist_norm/Ra_inter)*PI/Ra_inter*dist.x/dist_norm;
        der3bsupp[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k]=-0.5*sin(PI*dist_norm/Ra_inter)*PI/Ra_inter*dist.y/dist_norm;
        der3bsupp[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k]=-0.5*sin(PI*dist_norm/Ra_inter)*PI/Ra_inter*dist.z/dist_norm;
      }
      //STEP 1: filling radial descriptor with larger cutoff

      if (dist_norm<Rs_inter){
            descriptors[actual_pos+k]=coeffa_inter/pow(dist_norm/Rs_inter,pow_alpha)+coeffb_inter/pow(dist_norm/Rs_inter,pow_beta)+coeffc_inter;

            double der_cutoff=(-pow_alpha*coeffa_inter/pow(dist_norm/Rs_inter,pow_alpha+1.)/rs-pow_beta*coeffb_inter/pow(dist_norm/Rs_inter,pow_beta+1.)/Rs_inter);

            der2b[b*N*3*radial_buffer+i*3*radial_buffer+k]=der_cutoff*dist.x/dist_norm;
            der2b[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k]=der_cutoff*dist.y/dist_norm;
            der2b[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k]=der_cutoff*dist.z/dist_norm;
        }
        else{

      descriptors[actual_pos+k]=0.5*(cos(PI*dist_norm/Rc_inter)+1);
      der2b[b*N*3*radial_buffer+i*3*radial_buffer+k]=-0.5*sin(PI*dist_norm/Rc_inter)*PI/Rc_inter*dist.x/dist_norm;
      der2b[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k]=-0.5*sin(PI*dist_norm/Rc_inter)*PI/Rc_inter*dist.y/dist_norm;
      der2b[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k]=-0.5*sin(PI*dist_norm/Rc_inter)*PI/Rc_inter*dist.z/dist_norm;

      //STEP 2: filling interaction map for pair descriptors
      intmap2b[b*N*(radial_buffer+1)+i*(radial_buffer+1)+1+k]=with[b*radial_buffer*N+i*radial_buffer+k];
        }

    }
    }
  }

  __syncthreads();

  if (threadIdx.x==0)
  {
      for (int i=0;i<BLOCK_DIM;i++)
      {
        atomicAdd((int*)&(numtriplet[num_angolare[i].x*N+num_angolare[i].y]),num_angolare[i].z);
      }
  }

}

__global__ void DescriptorsAngular_kernel(double range,int radial_buffer,double range_angolare,int angular_buffer,int N,
                      double* position,const double* boxes,
                      int *howmany,int *with,
                      double* descriptors,int* intmap3b_l,
                      double* des3bsupp,double* der3b_l,
                      double* der3bsupp, int nf,int* numtriplet,int* map_intra_d,double Ra_inter)
{
  int t=blockIdx.x*blockDim.x+threadIdx.x;

  int2* intmap3b=(int2*)intmap3b_l;
  double2* der3b=(double2*)der3b_l;
  double3* coor=(double3*)position;
  // from t to b,par,j,k
  int b=t/(angular_buffer*N);
  int reminder=t%(angular_buffer*N);
  int i=reminder/angular_buffer;
  int nn=reminder%angular_buffer;
  if (t<nf*N*angular_buffer)
  {

    int na_dim=numtriplet[b*N+i];
    int na_tripl=na_dim*(na_dim-1)/2;
    if (nn<na_tripl){


    int j=0;
    int prev_row=0;
    int next_row=na_dim-j-1;
    while (nn>=next_row)
    {
        j+=1;
        prev_row=next_row;
        next_row+=na_dim-j-1;
    }
    int k=nn-prev_row+1+j;

    intmap3b[b*N*angular_buffer+i*angular_buffer+nn].x=with[b*N*radial_buffer+i*radial_buffer+j];
    intmap3b[b*N*angular_buffer+i*angular_buffer+nn].y=with[b*N*radial_buffer+i*radial_buffer+k];


    vector olddist;

    vector distj, distk;

    double dist_normj,dist_normk;

    int j_who=with[b*N*radial_buffer+i*radial_buffer+j];
    int k_who=with[b*N*radial_buffer+i*radial_buffer+k];

    int chtype_j=map_intra_d[i]-map_intra_d[j_who];
    int chtype_k=map_intra_d[i]-map_intra_d[k_who];

    olddist.x=coor[b*N+i].x-coor[b*N+j_who].x;
    olddist.y=coor[b*N+i].y-coor[b*N+j_who].y;
    olddist.z=coor[b*N+i].z-coor[b*N+j_who].z;

    olddist.x-=rint(olddist.x);
    olddist.y-=rint(olddist.y);
    olddist.z-=rint(olddist.z);

    distj.x=boxes[b*6+0]*olddist.x+boxes[b*6+1]*olddist.y+boxes[b*6+2]*olddist.z;
    distj.y=boxes[b*6+3]*olddist.y+boxes[b*6+4]*olddist.z;
    distj.z=boxes[b*6+5]*olddist.z;

    dist_normj=sqrt(SQR(distj.x)+SQR(distj.y)+SQR(distj.z));

    olddist.x=coor[b*N+i].x-coor[b*N+k_who].x;
    olddist.y=coor[b*N+i].y-coor[b*N+k_who].y;
    olddist.z=coor[b*N+i].z-coor[b*N+k_who].z;

    olddist.x-=rint(olddist.x);
    olddist.y-=rint(olddist.y);
    olddist.z-=rint(olddist.z);

    distk.x=boxes[b*6+0]*olddist.x+boxes[b*6+1]*olddist.y+boxes[b*6+2]*olddist.z;
    distk.y=boxes[b*6+3]*olddist.y+boxes[b*6+4]*olddist.z;
    distk.z=boxes[b*6+5]*olddist.z;

    dist_normk=sqrt(SQR(distk.x)+SQR(distk.y)+SQR(distk.z));
    double angle=(distj.x*distk.x+distj.y*distk.y+distj.z*distk.z)/(dist_normj*dist_normk);

    //Here we implement a cosine cutoff
    double cutoffj,cutoffk;
    double3 dcij,dcik;

    double range_angolare_j,range_angolare_k;

    if (chtype_j==0){
        range_angolare_j=range_angolare;
    }
    else{
        range_angolare_j=Ra_inter;
    }
    if (chtype_k==0){
        range_angolare_k=range_angolare;
    }
    else{
        range_angolare_k=Ra_inter;
    }

    if (dist_normj<range_angolare_j && dist_normk<range_angolare_k){

        cutoffj=0.5f*(1.f+cos(PI*dist_normj/range_angolare_j));
        cutoffk=0.5f*(1.f+cos(PI*dist_normk/range_angolare_k));

        double tijk=0.5*(angle+1)*cutoffj*cutoffk;


        dcij.x=-0.5f*sin(PI*dist_normj/range_angolare_j)*PI/range_angolare_j/dist_normj*distj.x;
        dcij.y=-0.5f*sin(PI*dist_normj/range_angolare_j)*PI/range_angolare_j/dist_normj*distj.y;
        dcij.z=-0.5f*sin(PI*dist_normj/range_angolare_j)*PI/range_angolare_j/dist_normj*distj.z;

        dcik.x=-0.5f*sin(PI*dist_normk/range_angolare_k)*PI/range_angolare_k/dist_normk*distk.x;
        dcik.y=-0.5f*sin(PI*dist_normk/range_angolare_k)*PI/range_angolare_k/dist_normk*distk.y;
        dcik.z=-0.5f*sin(PI*dist_normk/range_angolare_k)*PI/range_angolare_k/dist_normk*distk.z;


        double3 dangleij,dangleik;

        dangleij.x = 0.5 * (SQR(dist_normj) * distk.x - distj.x * (distj.x * distk.x + distj.y * distk.y + distj.z * distk.z)) / (dist_normj * dist_normj* dist_normj * dist_normk);
        dangleij.y = 0.5 * (SQR(dist_normj) * distk.y - distj.y * (distj.x * distk.x + distj.y * distk.y + distj.z * distk.z)) / (dist_normj * dist_normj* dist_normj * dist_normk);
        dangleij.z = 0.5 * (SQR(dist_normj) * distk.z - distj.z * (distj.x * distk.x + distj.y * distk.y + distj.z * distk.z)) / (dist_normj * dist_normj* dist_normj * dist_normk);

        dangleik.x = 0.5 * (SQR(dist_normk) * distj.x - distk.x * (distj.x * distk.x + distj.y * distk.y + distj.z * distk.z)) / (dist_normk * dist_normk* dist_normk * dist_normj);
        dangleik.y = 0.5 * (SQR(dist_normk) * distj.y - distk.y * (distj.x * distk.x + distj.y * distk.y + distj.z * distk.z)) / (dist_normk * dist_normk* dist_normk * dist_normj);
        dangleik.z = 0.5 * (SQR(dist_normk) * distj.z - distk.z * (distj.x * distk.x + distj.y * distk.y + distj.z * distk.z)) / (dist_normk * dist_normk* dist_normk * dist_normj);
    
    
    int na=angular_buffer;
    der3b[b*N*na*3+i*na*3+na*0+nn].x=dangleij.x*cutoffj*cutoffk+0.5*(angle+1)*dcij.x*cutoffk;
    der3b[b*N*na*3+i*na*3+na*0+nn].y=dangleik.x*cutoffj*cutoffk+0.5*(angle+1)*cutoffj*dcik.x;

    der3b[b*N*na*3+i*na*3+na*1+nn].x=dangleij.y*cutoffj*cutoffk+0.5*(angle+1)*dcij.y*cutoffk;
    der3b[b*N*na*3+i*na*3+na*1+nn].y=dangleik.y*cutoffj*cutoffk+0.5*(angle+1)*cutoffj*dcik.y;

    der3b[b*N*na*3+i*na*3+na*2+nn].x=dangleij.z*cutoffj*cutoffk+0.5*(angle+1)*dcij.z*cutoffk;
    der3b[b*N*na*3+i*na*3+na*2+nn].y=dangleik.z*cutoffj*cutoffk+0.5*(angle+1)*cutoffj*dcik.z;

    descriptors[b*N*angular_buffer+i*angular_buffer+nn]=tijk;
    }

     }

  }
}

void fill_radial_launcher(double R_c,int radbuff,double R_a,int angbuff,int N,
                      double* inopos_d,const double* box_d,
                      int *howmany_d,int *with_d,
                      double* descriptor_d,int* intmap2b_d,double* der2b_d,
                      double* des3bsupp_d,
                      double* der3bsupp_d, int nf,int* numtriplet_d,
                      double rs, double coeffa_intra,double coeffb_intra,double coeffc_intra,
                      double coeffa_inter, double coeffb_inter, double coeffc_inter,
                      double pow_alpha, double pow_beta,double Rs_inter,double Rc_inter,
                      const int* map_intra_d,double Ra_inter){

      dim3 dimGrid(ceil(double(nf*N*radbuff)/double(BLOCK_DIM)),1,1);
      dim3 dimBlock(BLOCK_DIM,1,1);

      TF_CHECK_OK(::tensorflow::GpuLaunchKernel(DescriptorsRadial_kernel,dimGrid, dimBlock,
                  0, nullptr,R_c,radbuff,R_a,angbuff,N,inopos_d,box_d,
                      howmany_d,with_d,descriptor_d,intmap2b_d,der2b_d,
                      des3bsupp_d,der3bsupp_d,nf,numtriplet_d,
                      rs,coeffa_intra,coeffb_intra,coeffc_intra,coeffa_inter,coeffb_inter,
                      coeffc_inter,pow_alpha,pow_beta,Rs_inter,
                      Rc_inter,map_intra_d,Ra_inter));
      cudaDeviceSynchronize();

}



void fill_angular_launcher(double R_c,int radbuff,double R_a,int angbuff,int N,
                      double* inopos_d,const double* box_d,
                      int *howmany_d,int *with_d,
                      double* ang_descr_d,int* intmap3b_d,
                      double* des3bsupp_d,double* der3b_d,
                      double* der3bsupp_d, int nf,int* numtriplet_d,
                      const int* map_intra_d,double Ra_inter){

                dim3 dimGrid(ceil(double(nf*N*angbuff)/double(BLOCK_DIM)),1,1);
                dim3 dimBlock(BLOCK_DIM,1,1);

                TF_CHECK_OK(::tensorflow::GpuLaunchKernel(DescriptorsAngular_kernel,
                           dimGrid,dimBlock,0, nullptr,R_c,radbuff,R_a,angbuff,N,
                           inopos_d,box_d,howmany_d,with_d,
                           ang_descr_d,intmap3b_d,des3bsupp_d,der3b_d,
                           der3bsupp_d,nf,numtriplet_d
                           ,map_intra_d,Ra_inter));

                cudaDeviceSynchronize();

     }

__global__ void set_tensor_to_zero_double_kernel(double* tensor,int dim){
          int t=blockIdx.x*blockDim.x+threadIdx.x;

          if (t<dim)
             tensor[t]=0.f;
}
void set_tensor_to_zero_double(double* tensor,int dimten){
     int grids=ceil(double(dimten)/double(300));
     dim3 dimGrid(grids,1,1);
     dim3 dimBlock(300,1,1);
     TF_CHECK_OK(::tensorflow::GpuLaunchKernel(set_tensor_to_zero_double_kernel,dimGrid,dimBlock, 0, nullptr,tensor,dimten));
     cudaDeviceSynchronize();
     }

__global__ void set_tensor_to_zero_int_kernel(int* tensor,int dim){
          int t=blockIdx.x*blockDim.x+threadIdx.x;

          if (t<dim)
             tensor[t]=0;
}
void set_tensor_to_zero_int(int* tensor,int dimten){
     int grids=ceil(double(dimten)/double(300));
     dim3 dimGrid(grids,1,1);
     dim3 dimBlock(300,1,1);
     TF_CHECK_OK(::tensorflow::GpuLaunchKernel(set_tensor_to_zero_int_kernel,dimGrid,dimBlock, 0, nullptr,tensor,dimten));
     cudaDeviceSynchronize();
     }
__global__ void check_max_kernel(int* tensor,int dim,int maxval,int* resval){
           int t=blockIdx.x*blockDim.x+threadIdx.x;
	   if (t<dim){
	      if (tensor[t]*(tensor[t]+1)/2>maxval){
                  atomicAdd((int*)&(resval[0]),tensor[t]);
	         }
           }
}
void check_max_launcher(int* tensor,int dimten,int maxval,int* resval){
     int grids=ceil(double(dimten)/double(300));
     dim3 dimGrid(grids,1,1);
     dim3 dimBlock(300,1,1);
     TF_CHECK_OK(::tensorflow::GpuLaunchKernel(check_max_kernel,dimGrid,dimBlock, 0, nullptr,tensor,dimten,maxval,resval));
     cudaDeviceSynchronize();
}
//#endif
