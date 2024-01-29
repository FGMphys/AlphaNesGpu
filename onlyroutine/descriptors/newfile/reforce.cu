#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <ctype.h>
#include <cuda.h>
#include <time.h>

#include "vector.h"
#include "interaction_map.h"
#include "cell_list.h"
#include "smart_allocator.h"
#include "utilities.h"
#include "log.h"


#define SQR(x) ((x)*(x))
#define Sqrt(x) (sqrt(x))
#define Power(x,n) (powf(x,n))

#define MAX_LINE_LENGTH 2000
#define MAX_NNEIGHBOURS 300

#define BLOCK_DIM 256

#define PI 3.141592654f

typedef struct _distsymm {
  int index;
  float dist;
  float dx;
  float dy;
  float dz;
} distsymm;

typedef struct _distangle {
  int indexj;
  int indexk;
  float distj;
  float distk;
  float angle;
  float dxij;
  float dyij;
  float dzij;
  float dxik;
  float dyik;
  float dzik;
} distangle;



// global variables
static listcell *cells;
static int N;
static float box[6],inobox[6];
static interactionmap *ime;


static int Radbuff,Angbuff;

static float R_c,Rs,coeffA,coeffB,coeffC,Pow_alpha,Pow_beta;

int getLine(char *line,FILE *pfile)
{
        if (fgets(line,MAX_LINE_LENGTH,pfile) == NULL)
                return 0;
        else
                return strlen(line);
}

void read_data_float(char* file_path,float* tbw){
     char line[100]="";
     char filename[100];
     sprintf(filename,"%s",file_path);

     FILE* ptr;
     ptr = fopen(file_path, "r");
     printf("Open file %s\n",filename);
    if (NULL == ptr) {
        printf("file can't be opened \n");
    }
    int k=0;
    while (getLine(line,ptr)!=0)
        {
        sscanf(line,"%f", tbw+k);
        k=k+1;
    }
    fclose(ptr);
     }



void save_cutoff(float rc){
  FILE *newfile;
  newfile=fopen("cutoff_curve.dat","w");
  float dx=rc/1000.;
  float x=0;
  for (int k=0;k<1000;k++){
    x=x+dx;
    if (x<Rs){
      fprintf(newfile,"%g %g\n",x,coeffA/Power(x,Pow_alpha)+coeffB/Power(x,Pow_beta)+coeffC);
    }
    else{
      fprintf(newfile,"%g %g\n",x,0.5*(1+cos(PI*x/rc)));
  }
}
   fclose(newfile);
}
void construct_repulsion(float rc){
    float alpha=1.;
    float beta=-30.;
    Pow_alpha=alpha;
    Pow_beta=beta;
    float rs=Rs;
    float f=0.5*(cos(PI*rs/rc)+1);
    float f1=-0.5*PI/rc*sin(PI*rs/rc);
    float f2_red=-0.5*SQR(PI/rc)*cos(PI*rs/rc)*SQR(rs);
    float gamma_red=1./(alpha-beta)*alpha-Power(rs,alpha);
    float delta_red=1./(alpha-beta)*(f*(alpha-beta)-f1*rs-f*alpha);
    float eta_red=-alpha/(alpha-beta);
    float epsilon_red=1./(alpha-beta)*(rs*f1+alpha*f);
    float c2_red=alpha*(alpha+1)*delta_red+beta*(beta+1)*epsilon_red;
    float c1_red=alpha*(alpha+1)*gamma_red+beta*(beta+1)*eta_red;
    coeffC=(f2_red-c2_red)/c1_red;
    float eta=-alpha*Power(rs,beta)/(alpha-beta);
    float epsilon=Power(rs,beta)/(alpha-beta)*(rs*f1+alpha*f);
    coeffB=eta*coeffC+epsilon;
    float gamma=Power(rs,alpha)/(alpha-beta)*alpha-Power(rs,alpha);
    float delta=Power(rs,alpha)/(alpha-beta)*(f*(alpha-beta)-f1*rs-f*alpha);
    coeffA=gamma*coeffC+delta;
    save_cutoff(rc);

}


 void construct_descriptor(float* box,float rc,int N,
                           int rad_buffer,int angular_buffer){

          inobox[0]=1./box[0];
          inobox[1]=-box[1]/(box[0]*box[3]);
          inobox[2]=(box[1]*box[4])/(box[0]*box[3]*box[5])-box[2]/(box[0]*box[5]);
          inobox[3]=1./box[3];
          inobox[4]=-box[4]/(box[3]*box[5]);
          inobox[5]=1./box[5];

          R_c=rc;

          cells=getList(box,rc,N);

          // INTERACTION MAPS
          ime=createInteractionMap(N,rad_buffer);


          Radbuff=rad_buffer;
	  Angbuff=angular_buffer;
	  construct_repulsion(rc);
 }
__global__ void DescriptorsRadial_kernel(float range,int radial_buffer,float range_angolare,int angular_buffer,int N,
                      float* position,float* boxes,
                      int *howmany,int *with,
                      float* descriptors,int* intmap2b,float* der2b,
                      float* des3bsupp,
                      float* der3bsupp, int nf,int* numtriplet,
		      float rs, float coeffa,float coeffb,float coeffc,float pow_alpha, float pow_beta)
{
  int t=blockIdx.x*blockDim.x+threadIdx.x;

  // from t to b,par,j,k
  int b=t/(radial_buffer*N);
  int reminder=t%(radial_buffer*N);
  int i=reminder/radial_buffer;
  int k=reminder%radial_buffer;

  float3* coor=(float3*)position;
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
    float dist_norm;

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
      //STEP 0: filling the radial part with the smaller angular cut-off
      if (dist_norm<range_angolare)
      {
        num_angolare[threadIdx.x].z=1;

        des3bsupp[actual_pos+k]=0.5*(cosf(PI*dist_norm/range_angolare)+1);
        der3bsupp[b*N*3*radial_buffer+i*3*radial_buffer+k]=-0.5*sinf(PI*dist_norm/range_angolare)*PI/range_angolare*dist.x/dist_norm;
        der3bsupp[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k]=-0.5*sinf(PI*dist_norm/range_angolare)*PI/range_angolare*dist.y/dist_norm;
        der3bsupp[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k]=-0.5*sinf(PI*dist_norm/range_angolare)*PI/range_angolare*dist.z/dist_norm;
      }
      //STEP 1: filling radial descriptor with larger cutoff
      if (dist_norm<rs){
          descriptors[actual_pos+k]=coeffa/Power(dist_norm,pow_alpha)+coeffb/Power(dist_norm,pow_beta)+coeffc;

          der2b[b*N*3*radial_buffer+i*3*radial_buffer+k]=(-pow_alpha*coeffa/Power(dist_norm,pow_alpha+1.)-pow_beta*coeffb/Power(dist_norm,pow_beta+1.))*dist.x/dist_norm;
          der2b[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k]=(-pow_alpha*coeffa/Power(dist_norm,pow_alpha+1.)-pow_beta*coeffb/Power(dist_norm,pow_beta+1.))*dist.y/dist_norm;
          der2b[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k]=(-pow_alpha*coeffa/Power(dist_norm,pow_alpha+1.)-pow_beta*coeffb/Power(dist_norm,pow_beta+1.))*dist.z/dist_norm;
      }
      else{
        descriptors[actual_pos+k]=0.5*(cosf(PI*dist_norm/range)+1);
        der2b[b*N*3*radial_buffer+i*3*radial_buffer+k]=-0.5*sinf(PI*dist_norm/range)*PI/range*dist.x/dist_norm;
        der2b[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k]=-0.5*sinf(PI*dist_norm/range)*PI/range*dist.y/dist_norm;
        der2b[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k]=-0.5*sinf(PI*dist_norm/range)*PI/range*dist.z/dist_norm;
      }
      //STEP 2: filling interaction map for pair descriptors
      intmap2b[b*N*(radial_buffer+1)+i*(radial_buffer+1)+1+k]=with[b*radial_buffer*N+i*radial_buffer+k];

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

__global__ void DescriptorsAngular_kernel(float range,int radial_buffer,float range_angolare,int angular_buffer,int N,
                      float* position,float* boxes,
                      int *howmany,int *with,
                      float* descriptors,int* intmap3b_l,
                      float* des3bsupp,float* der3b_l,
                      float* der3bsupp, int nf,int* numtriplet)
{
  int t=blockIdx.x*blockDim.x+threadIdx.x;

  int2* intmap3b=(int2*)intmap3b_l;
  float2* der3b=(float2*)der3b_l;
  float3* coor=(float3*)position;
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

    intmap3b[b*N*angular_buffer+i*angular_buffer].x=with[b*N*radial_buffer+i*radial_buffer+j];
    intmap3b[b*N*angular_buffer+i*angular_buffer].y=with[b*N*radial_buffer+i*radial_buffer+k];



    vector olddist;

    vector distj, distk;

    float dist_normj,dist_normk;

    int j_who=with[b*N*radial_buffer+i*radial_buffer+j];
    int k_who=with[b*N*radial_buffer+i*radial_buffer+k];

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
    float angle=(distj.x*distk.x+distj.y*distk.y+distj.z*distk.z)/(dist_normj*dist_normk);

    //Here we implement a cosine cutoff
    float cutoffj,cutoffk;
    float3 dcij,dcik;

    cutoffj=0.5f*(1.f+cosf(PI*dist_normj/range_angolare));
    cutoffk=0.5f*(1.f+cosf(PI*dist_normk/range_angolare));

    float tijk=0.5*(angle+1)*cutoffj*cutoffk;


    dcij.x=-0.5f*sinf(PI*dist_normj/range_angolare)*PI/range_angolare/dist_normj*distj.x;
    dcij.y=-0.5f*sinf(PI*dist_normj/range_angolare)*PI/range_angolare/dist_normj*distj.y;
    dcij.z=-0.5f*sinf(PI*dist_normj/range_angolare)*PI/range_angolare/dist_normj*distj.z;

    dcik.x=-0.5f*sinf(PI*dist_normk/range_angolare)*PI/range_angolare/dist_normk*distk.x;
    dcik.y=-0.5f*sinf(PI*dist_normk/range_angolare)*PI/range_angolare/dist_normk*distk.y;
    dcik.z=-0.5f*sinf(PI*dist_normk/range_angolare)*PI/range_angolare/dist_normk*distk.z;


    float3 dangleij,dangleik;

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

void fill_radial_launcher(float R_c,int radbuff,float R_a,int angbuff,int N,
                      float* inopos_d,float* box_d,
                      int *howmany_d,int *with_d,
                      float* descriptor_d,int* intmap2b_d,float* der2b_d,
                      float* des3bsupp_d,
                      float* der3bsupp_d, int nf,int* numtriplet_d,
                      float rs, float coeffa,float coeffb,float coeffc,float pow_alpha, float pow_beta){


      dim3 dimGrid(ceil(float(nf*N*radbuff)/float(BLOCK_DIM)),1,1);
      dim3 dimBlock(BLOCK_DIM,1,1);

      DescriptorsRadial_kernel<<<dimGrid,dimBlock>>>(R_c,radbuff,R_a,angbuff,N,
                      inopos_d,box_d,
                      howmany_d,with_d,
                      descriptor_d,intmap2b_d,der2b_d,
                      des3bsupp_d,
                      der3bsupp_d,nf,numtriplet_d,
                      rs,coeffa,coeffb,coeffc,pow_alpha,pow_beta);
      cudaDeviceSynchronize();


}



void fill_angular_launcher(float R_c,int radbuff,float R_a,int angbuff,int N,
                      float* inopos_d,float* box_d,
                      int *howmany_d,int *with_d,
                      float* ang_descr_d,int* intmap3b_d,
                      float* des3bsupp_d,float* der3b_d,
                      float* der3bsupp_d, int nf,int* numtriplet_d){

                dim3 dimGrid(ceil(float(nf*N*angbuff)/float(BLOCK_DIM)),1,1);
                dim3 dimBlock(BLOCK_DIM,1,1);
	        printf("Ok before kernel\n");

                DescriptorsAngular_kernel<<<dimGrid,dimBlock>>>(R_c,radbuff,R_a,angbuff,N,
                      inopos_d,box_d,
                      howmany_d,with_d,
                      ang_descr_d,intmap3b_d,
                      des3bsupp_d,der3b_d,
                      der3bsupp_d,nf,numtriplet_d);

                cudaDeviceSynchronize();

     }

int main(int argv,char** argc){
	clock_t start_celle,fine_celle,start_mappa;
    float r_c=4.5;
    float R_a=4.5;
    Rs=2.25;
    N=768;
    int b=1; //number of frames
    int nf=b;
    int radbuff=80;
    int angbuff=1378;
    //Reading position and box
    float* nowbox=(float*)calloc(b*6,sizeof(float));
    float* nowinobox=(float*)calloc(b*6,sizeof(float));
    float* nowpos=(float*)calloc(b*N*3,sizeof(float));
    char* input_name_box={"serial_benchmark/input_data/box_4debug_flat"};
    read_data_float(input_name_box,nowbox);
    read_data_float("serial_benchmark/input_data/pos_4debug_flat",nowpos);

    int *howmany_d;
    int *with_d;
    cudaMalloc(&howmany_d,nf*N*sizeof(int));
    cudaMalloc(&with_d,nf*N*radbuff*sizeof(int));


    //Construct descriptor utilities
    construct_descriptor(nowbox,r_c,N,
                           radbuff,angbuff);


    // loop per il calcolo di inobox, posizioni in spazio reale, e mappe di vicini per tutti i frame
    int ii;
    for (ii=0;ii<nf;ii++)
    {
      nowinobox[ii*6+0]=1./nowbox[ii*6+0];
      nowinobox[ii*6+1]=-nowbox[ii*6+1]/(nowbox[ii*6+0]*nowbox[ii*6+3]);
      nowinobox[ii*6+2]=(nowbox[ii*6+1]*nowbox[ii*6+4])/(nowbox[ii*6+0]*nowbox[ii*6+3]*nowbox[ii*6+5])-nowbox[ii*6+2]/(nowbox[ii*6+0]*nowbox[ii*6+5]);
      nowinobox[ii*6+3]=1./nowbox[ii*6+3];
      nowinobox[ii*6+4]=-nowbox[ii*6+4]/(nowbox[ii*6+3]*nowbox[ii*6+5]);
      nowinobox[ii*6+5]=1./nowbox[ii*6+5];

      for (int i=0;i<N;i++){
        float px=nowpos[ii*N*3+i*3];
        float py=nowpos[ii*N*3+i*3+1];
        float pz=nowpos[ii*N*3+i*3+2];

        nowpos[ii*N*3+i*3]=(inobox[ii*6+0]*px+inobox[ii*6+1]*py+inobox[ii*6+2]*pz);
        nowpos[ii*N*3+i*3+1]=(inobox[ii*6+3]*py+inobox[ii*6+4]*pz);
        nowpos[ii*N*3+i*3+2]=(inobox[ii*6+5]*pz);
      }

      // calcolo delle celle e dei neighbour list
      start_celle=clock();
      fullUpdateList(cells,pos,N,&nowbox[ii*6],R_c);
      start_mappa=clock();
      resetInteractionMap(ime);
      calculateInteractionMapWithCutoffDistanceOrdered(cells,ime,pos,&nowbox[ii*6],R_c);
      fine_celle=clock();



      cudaMemcpy(howmany_d+ii*N,ime->howmany,N*sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(with_d+ii*N*radbuff,ime->with[0],N*radbuff*sizeof(int),cudaMemcpyHostToDevice);
      for (int i=0;i<N;i++)
      {
        if (ime->howmany[i]>Radbuff)
        {
          printf("Buffer radiale saturato by \n");
	  printf("Particle %d at frame %d with %d neighbours \n",i,ii,ime->howmany[i]);
          exit(1);
        }
      }

    }


    float *box_d;
    float *inobox_d;
    cudaMalloc(&box_d,nf*6*sizeof(float));
    cudaMalloc(&inobox_d,nf*6*sizeof(float));

    cudaMemcpy(box_d,nowbox,nf*6*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(inobox_d,nowinobox,nf*6*sizeof(float),cudaMemcpyHostToDevice);

    float *inopos_d;
    cudaMalloc(&inopos_d,nf*N*3*sizeof(float));
    cudaMemcpy(inopos_d,nowpos,nf*N*3*sizeof(float),cudaMemcpyHostToDevice);


    float* rad_descr_d;
    int* intmap2b_d;
    float* der2b_d;

    float* ang_descr_d;
    int* intmap3b_d;
    float* des3bsupp_d;
    float* der3b_d;
    float* der3bsupp_d;
    int* numtriplet_d;

    // Allocate device memory
    cudaMalloc((void**)&rad_descr_d, b * N * (radbuff + angbuff) * sizeof(float));
    cudaMalloc((void**)&intmap2b_d, b * N * (radbuff + 1) * sizeof(int));
    cudaMalloc((void**)&der2b_d, b * N * (radbuff * 3) * sizeof(float));

    cudaMalloc((void**)&ang_descr_d, b * N * angbuff * sizeof(float));
    cudaMalloc((void**)&intmap3b_d, b * N * (angbuff * 2) * sizeof(int));
    cudaMalloc((void**)&des3bsupp_d, b * N * (radbuff) * sizeof(float));
    cudaMalloc((void**)&der3b_d, b * N * (angbuff * 2 * 3) * sizeof(float));
    cudaMalloc((void**)&der3bsupp_d, b * N * (radbuff * 3) * sizeof(float));
    cudaMalloc((void**)&numtriplet_d, b * N * sizeof(int));

    // Initialize allocated memory to zero
    cudaMemset(rad_descr_d, 0, b * N * (radbuff) * sizeof(float));
    cudaMemset(intmap2b_d, 0, b * N * (radbuff + 1) * sizeof(int));
    cudaMemset(der2b_d, 0, b * N * (radbuff * 3) * sizeof(float));


    cudaMemset(ang_descr_d, 0, b * N * angbuff * sizeof(int));
    cudaMemset(intmap3b_d, 0, b * N * (angbuff * 2) * sizeof(int));
    cudaMemset(des3bsupp_d, 0, b * N * (radbuff) * sizeof(float));
    cudaMemset(der3b_d, 0, b * N * (angbuff * 2 * 3) * sizeof(float));
    cudaMemset(der3bsupp_d, 0, b * N * (radbuff * 3) * sizeof(float));
    cudaMemset(numtriplet_d, 0, b * N * sizeof(int));

    printf("Ok before launcher\n");

    clock_t start_kernel=clock();

    fill_radial_launcher(R_c,radbuff,R_a,angbuff,N,
                      inopos_d,box_d,
                      howmany_d,with_d,
                      rad_descr_d,intmap2b_d,der2b_d,
                      des3bsupp_d,
                      der3bsupp_d,nf,numtriplet_d,
                      Rs,coeffA,coeffB,coeffC,Pow_alpha,Pow_beta);
    fill_angular_launcher(R_c, radbuff, R_a, angbuff, N, inopos_d,
		         box_d, howmany_d, with_d, ang_descr_d,
			 intmap3b_d, des3bsupp_d, der3b_d, der3bsupp_d,
			 nf, numtriplet_d);

    clock_t fine_kernel=clock();

    printf("start celle %ld fine celle %ld start_kernel %ld fine_kernel %ld rapporto: %lf rapporto2 %lf\n",start_celle,fine_celle,start_kernel,fine_kernel,(double)(fine_celle-start_celle)/(double)(fine_kernel-start_kernel),(double)(fine_celle-start_mappa)/(double)(fine_celle-start_celle));

    float* rad_descr=(float*)calloc(nf*N*radbuff,sizeof(float));
    float* ang_descr=(float*)calloc(nf*N*angbuff,sizeof(float));
    float* der3b=(float*)calloc(2*3*nf*N*angbuff,sizeof(float));

    cudaMemcpy(rad_descr,rad_descr_d,nf*N*radbuff*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(ang_descr,ang_descr_d,nf*N*angbuff*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(der3b,der3b_d,2*3*nf*N*angbuff*sizeof(float),cudaMemcpyDeviceToHost);
    printf("First rad descr %g\n",rad_descr[nf*(N-1)*radbuff]);
    printf("First ang descr %g\n",ang_descr[nf*(256-1)*angbuff]);
    printf("First ang descr %g\n",ang_descr[angbuff]);
    printf("First ang descr %g %g\n",der3b[0],der3b[2*3*nf*(N-1)*angbuff]);

    return 0;
}
