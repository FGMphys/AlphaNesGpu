#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <ctype.h>
#include <cuda.h>


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

#define BLOCK_DIM 400

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
static vector* pos;
static interactionmap *im,*ime;

static distsymm **ds;
static distangle **da;
static int *ds_num;
static int *da_num;
static int *ds_num_angular;

static int n_input;
static int n_input_angolare;

static float Rs,coeffA,coeffB,coeffC,Pow_alpha,Pow_beta;

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
  float pi=3.1415926535;
  float x=0;
  for (int k=0;k<1000;k++){
    x=x+dx;
    if (x<Rs){
      fprintf(newfile,"%g %g\n",x,coeffA/Power(x,Pow_alpha)+coeffB/Power(x,Pow_beta)+coeffC);
    }
    else{
      fprintf(newfile,"%g %g\n",x,0.5*(1+cos(pi*x/rc)));
  }
}
   fclose(newfile);
}
void construct_repulsion(float rc){
    float alpha=1.;
    float beta=-30.;
    float pi=3.1415926535;
    Pow_alpha=alpha;
    Pow_beta=beta;
    float rs=Rs;
    float f=0.5*(cos(pi*rs/rc)+1);
    float f1=-0.5*pi/rc*sin(pi*rs/rc);
    float f2_red=-0.5*SQR(pi/rc)*cos(pi*rs/rc)*SQR(rs);
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





int notcompare(const void * a, const void * b)
{
  return -1;
}


int compare_distsymm(const void * a, const void * b)
{

  if ( ((distsymm*)a)->dist <  ((distsymm*)b)->dist )
    return -1;
  else
    return 1;

}



int insertionSort_distsymm(distsymm *array,distsymm *element,distsymm *buffer,int *length,size_t size,int (*compar)(const void*,const void*))
{
	int l=*length;

  memcpy(array+l,element,size);

	while ((l>0) && (compar(array+(l-1),array+l)==1))
	{
    memcpy(buffer,array+l,size);
    memcpy(array+l,array+(l-1),size);
    memcpy(array+(l-1),buffer,size);

		l--;
	}
	(*length)++;

  return l;
}

int insertionSort_distangle(distangle *array,distangle *element,distangle *buffer,int *length,size_t size,int (*compar)(const void*,const void*))
{
	int l=*length;

  memcpy(array+l,element,size);

	while ((l>0) && (compar(array+(l-1),array+l)==1))
	{
    memcpy(buffer,array+l,size);
    memcpy(array+l,array+(l-1),size);
    memcpy(array+(l-1),buffer,size);

		l--;
	}
	(*length)++;

  return l;
}


 void construct_descriptor(float* box,float rc,int N,
                           int rad_buffer,int angular_buffer){

          inobox[0]=1./box[0];
          inobox[1]=-box[1]/(box[0]*box[3]);
          inobox[2]=(box[1]*box[4])/(box[0]*box[3]*box[5])-box[2]/(box[0]*box[5]);
          inobox[3]=1./box[3];
          inobox[4]=-box[4]/(box[3]*box[5]);
          inobox[5]=1./box[5];



          cells=getList(box,rc,N);

          // INTERACTION MAPS
          ime=createInteractionMap(N,MAX_NNEIGHBOURS);
          im=createInteractionMap(N,MAX_NNEIGHBOURS);

          pos=(vector*) calloc(N,sizeof(vector));

          //Allocazione per calcolo dei descrittori
          ds=(distsymm**)calloc(N,sizeof(distsymm*));
          da=(distangle**)calloc(N,sizeof(distangle*));
          ds_num=(int*)calloc(N,sizeof(int));
          da_num=(int*)calloc(N,sizeof(int));
          ds_num_angular=(int*)calloc(N,sizeof(int));


          n_input=rad_buffer;
          n_input_angolare=angular_buffer;
          for (int i=0;i<N;i++)
          {
            ds[i]=(distsymm*)calloc(n_input,sizeof(distsymm));
            da[i]=(distangle*)calloc(n_input_angolare,sizeof(distangle));
            ds_num[i]=0;
            ds_num_angular[i]=0;
            da_num[i]=0;
          }

    }


__global__ void computeDescriptorsRadial(float range,int radial_buffer,float range_angolare,int angular_buffer,int N,
                      float* pos,float* boxes,float *inoboxes,
                      int *howmany,int *with,
                      float* descriptors,int* intmap2b,int* intmap3b,
                      float* des3bsupp,float* der3b,float* der2b,
                      float* der3bsupp, int number_of_frames,int* numtriplet)
{
  int t=blockIdx.x*blockDim.x+threadIdx.x;

  // from t to b,par,j,k
  int b=t/(radial_buffer*N);
  int reminder=t%(radial_buffer*N);
  int i=reminder/radial_buffer;
  int k=reminder%radial_buffer;

  // shared memory counter for numero di vicini angolari
  __shared__ int3 num_angolare[BLOCK_DIM];

  num_angolare[threadIdx.x].x=b;
  num_angolare[threadIdx.x].y=i;
  num_angolare[threadIdx.x].z=0;


  if (t<b*N*radial_buffer)
  {

    if (k==0)
    {
      intmap2b[b*N*(radial_buffer+1)+i*(radial_buffer+1)]=howmany[b*N+i];
    }

    int ds_num_max=0;
    distsymm dij,buffer;
    vector olddist,dist;
    double dist_norm;
    int pos_index;

    if (k<howmany[b*N+i])
    {

      int j=with[b*N*radial_buffer+i*radial_buffer+k];
      // calcolo per le derivate
      olddist.x=pos[i].x-pos[j].x;
      olddist.y=pos[i].y-pos[j].y;
      olddist.z=pos[i].z-pos[j].z;

      olddist.x-=rint(olddist.x);
      olddist.y-=rint(olddist.y);
      olddist.z-=rint(olddist.z);

      dist.x=box[0]*olddist.x+box[1]*olddist.y+box[2]*olddist.z;
      dist.y=box[3]*olddist.y+box[4]*olddist.z;
      dist.z=box[5]*olddist.z;

      dist_norm=sqrt(SQR(dist.x)+SQR(dist.y)+SQR(dist.z));

      int actual_pos=b*N*radial_buffer+i*radial_buffer;

      if (dist_norm<range_angolare)
      {
        num_angolare[threadIdx.x].y=1;

        des3bsupp[actual_pos+k]=0.5*(cosf(pi*dist_norm/range_angolare)+1)
        der3bsupp[b*N*3*radial_buffer+i*3*radial_buffer+k]=-0.5*sinf(pi*dist_norm/range_angolare)*pi/range_angolare*dist.x/dist_norm;
        der3bsupp[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k]=-0.5*sinf(pi*dist_norm/range_angolare)*pi/range_angolare*dist.y/dist_norm;
        der3bsupp[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k]=-0.5*sinf(pi*dist_norm/range_angolare)*pi/range_angolare*dist.z/dist_norm;
      }





      if (dist<Rs){
          descriptors[actual_pos+k]=coeffA/Power(dist_norm,Pow_alpha)+coeffB/Power(dist_norm,Pow_beta)+coeffC;

          der2b[b*N*3*radial_buffer+i*3*radial_buffer+k]=(-Pow_alpha*coeffA/Power(dist_norm,Pow_alpha+1.)-Pow_beta*coeffB/Power(dist_norm,Pow_beta+1.))*dist.x/dist_norm;
          der2b[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k]=(-Pow_alpha*coeffA/Power(dist_norm,Pow_alpha+1.)-Pow_beta*coeffB/Power(dist_norm,Pow_beta+1.))*dist.y/dist_norm;
          der2b[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k]=(-Pow_alpha*coeffA/Power(dist_norm,Pow_alpha+1.)-Pow_beta*coeffB/Power(dist_norm,Pow_beta+1.))*dist.z/dist_norm;
      }
      else{
        descriptors[actual_pos+k]=0.5*(cosf(pi*dist_norm/range)+1);
        der2b[b*N*3*radial_buffer+i*3*radial_buffer+k]=-0.5*sinf(pi*dist_norm/range)*pi/range*dist.x/dist_norm;
        der2b[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k]=-0.5*sinf(pi*dist_norm/range)*pi/range*dist.y/dist_norm;
        der2b[b*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k]=-0.5*sinf(pi*dist_norm/range)*pi/range*dist.z/dist_norm;
      }

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

__global__ void computeDescriptorsAngular(float range,int radial_buffer,float range_angolare,int angular_buffer,int N,
                      float* position,float* boxes,float *inoboxes,
                      int *howmany,int *with,
                      float* descriptors,int* intmap2b,int* intmap3b_l,
                      float* des3bsupp,float* der3b_l,float* der2b,
                      float* der3bsupp, int number_of_frames,int* numtriplet)
{
  int t=blockIdx.x*blockDim.x+threadIdx.x;

  int2* intmap3b=(int2*)intmap3b_l;
  float2* der3b=(float2*)der3b_l;

  // from t to b,par,j,k
  int b=t/(angular_buffer*N);
  int reminder=t%(angular_buffer*N);
  int i=reminder/angular_buffer;
  int nn=reminder%angular_buffer;

  if (t<b*N*angular_buffer)
  {
    //CON NINPUT PARI A
    float pi=3.1415926535;

    int na_dim=numtriplet[b*N+i];//floorf(0.5f + sqrtf(0.25f + 2*na));
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

    olddist.x=pos[i].x-pos[j].x;
    olddist.y=pos[i].y-pos[j].y;
    olddist.z=pos[i].z-pos[j].z;

    olddist.x-=rint(olddist.x);
    olddist.y-=rint(olddist.y);
    olddist.z-=rint(olddist.z);

    distj.x=box[0]*olddist.x+box[1]*olddist.y+box[2]*olddist.z;
    distj.y=box[3]*olddist.y+box[4]*olddist.z;
    distj.z=box[5]*olddist.z;

    dist_normj=sqrt(SQR(dist.x)+SQR(dist.y)+SQR(dist.z));

    olddist.x=pos[i].x-pos[k].x;
    olddist.y=pos[i].y-pos[k].y;
    olddist.z=pos[i].z-pos[k].z;

    olddist.x-=rint(olddist.x);
    olddist.y-=rint(olddist.y);
    olddist.z-=rint(olddist.z);

    distk.x=box[0]*olddist.x+box[1]*olddist.y+box[2]*olddist.z;
    distk.y=box[3]*olddist.y+box[4]*olddist.z;
    distk.z=box[5]*olddist.z;

    dist_normk=sqrt(SQR(dist.x)+SQR(dist.y)+SQR(dist.z));
    float angle=(distj.x*distj.x+distj.y*distj.y+distj.z*distj.z)/(dist_normj*dist_normk);

	  //Here we implement a cosine cutoff
    float cutoffj,dcij;
    cutoffj=0.5f*(1.f+cosf(pi*dist_normj/range_angolare));
    dcij=-0.5f*sinf(pi*dist_normj/range_angolare)*pi/range_angolare/dist_normj;
    float cutoffk,dcik;
    cutoffk=0.5f*(1.f+cosf(pi*dist_normk/range_angolare));
    dcik=-0.5f*sinf(pi*dist_normk/range_angolare)*pi/range_angolare/dist_normk;
    float tijk=0.5*(angle+1)*cutoffj*cutoffk;

    float3 dcij,dcik;

    dcij.x=-0.5f*sinf(pi*dist_normj/range_angolare)*pi/range_angolare/dist_normj*distj.x;
    dcij.y=-0.5f*sinf(pi*dist_normj/range_angolare)*pi/range_angolare/dist_normj*distj.y;
    dcij.z=-0.5f*sinf(pi*dist_normj/range_angolare)*pi/range_angolare/dist_normj*distj.z;

    dcik.x=-0.5f*sinf(pi*dist_normk/range_angolare)*pi/range_angolare/dist_normk*distk.x;
    dcik.y=-0.5f*sinf(pi*dist_normk/range_angolare)*pi/range_angolare/dist_normk*distk.y;
    dcik.z=-0.5f*sinf(pi*dist_normk/range_angolare)*pi/range_angolare/dist_normk*distk.z;

    /*
    float danglexij=0.5*(dij2*xik - xij*(xij*xik + yij*yik + zij*zik))/(Power(dij2,1.5)*Sqrt(dik2));
    float dangleyij=0.5*(dij2*yik - yij*(xij*xik + yij*yik + zij*zik))/(Power(dij2,1.5)*Sqrt(dik2));
    float danglezij=0.5*(dij2*zik - zij*(xij*xik + yij*yik + zij*zik))/(Power(dij2,1.5)*Sqrt(dik2));

    float danglexik=0.5*(dik2*xij - xik*(xij*xik + yij*yik + zij*zik))/(Sqrt(dij2)*Power(dik2,1.5));
    float dangleyik=0.5*(dik2*yij - yik*(xij*xik + yij*yik + zij*zik))/(Sqrt(dij2)*Power(dik2,1.5));
    float danglezik=0.5*(dik2*zij - zik*(xij*xik + yij*yik + zij*zik))/(Sqrt(dij2)*Power(dik2,1.5));
    */

    float3 dangleij,dangleik;

    dangleij.x = 0.5 * (SQR(dist_normj) * distk.x - distj.x * (distj.x * distk.x + distj.y * distk.y + distj.z * distk.z)) / (dist_normj * dist_normj* dist_normj * dist_normk);
    dangleij.y = 0.5 * (SQR(dist_normj) * distk.y - distj.y * (distj.x * distk.x + distj.y * distk.y + distj.z * distk.z)) / (dist_normj * dist_normj* dist_normj * dist_normk);
    dangleij.z = 0.5 * (SQR(dist_normj) * distk.z - distj.z * (distj.x * distk.x + distj.y * distk.y + distj.z * distk.z)) / (dist_normj * dist_normj* dist_normj * dist_normk);

    dangleik.x = 0.5 * (SQR(dist_normk) * distj.x - distk.x * (distj.x * distk.x + distj.y * distk.y + distj.z * distk.z)) / (dist_normk * dist_normk* dist_normk * dist_normj);
    dangleik.y = 0.5 * (SQR(dist_normk) * distj.y - distk.y * (distj.x * distk.x + distj.y * distk.y + distj.z * distk.z)) / (dist_normk * dist_normk* dist_normk * dist_normj);
    dangleik.z = 0.5 * (SQR(dist_normk) * distj.z - distk.z * (distj.x * distk.x + distj.y * distk.y + distj.z * distk.z)) / (dist_normk * dist_normk* dist_normk * dist_normj);


    der3b[b*N*na*3+i*na*3+na*0+nn].x=dangleij.x*cutoffj*cutoffk+0.5*(angle+1)*dcij.x*cutoffk;
    der3b[b*N*na*3+i*na*3+na*0+nn].y=dangleik.x*cutoffj*cutoffk+0.5*(angle+1)*cutoffj*dcik.x;

    der3b[b*N*na*3+i*na*3+na*1+nn].x=dangleij.y*cutoffj*cutoffk+0.5*(angle+1)*dcij.y*cutoffk;
    der3b[b*N*na*3+i*na*3+na*1+nn].y=dangleik.y*cutoffj*cutoffk+0.5*(angle+1)*cutoffj*dcik.y;

    der3b[b*N*na*3+i*na*3+na*2+nn].x=dangleij.z*cutoffj*cutoffk+0.5*(angle+1)*dcij.z*cutoffk;
    der3b[b*N*na*3+i*na*3+na*2+nn].y=dangleik.z*cutoffj*cutoffk+0.5*(angle+1)*cutoffj*dcik.z;

    descriptors[b*N*angular_buffer+i*angular_buffer+nn]=tijk;


  }

}



void make_descriptors(float rc,int rad_buff,float rc_ang,int ang_buff,int N,
                      float* position,float* boxes,
                      float* descriptors,int* intmap2b,int* intmap3b,
                      float* des3bsupp,float* der3b,float* der2b,
                      float* der3bsupp, int number_of_frames,int* numtriplet)
{


  float range=rc;
  float range_angolare=rc_ang;
  int radial_buffer=rad_buff;
  int angular_buffer=ang_buff;

  for (int h=0;h<number_of_frames*N*(rad_buff+1);h++){
      intmap2b[h]=0;
  }

  for (int h=0;h<number_of_frames*N*(ang_buff*2);h++){
      intmap3b[h]=0;
  }
  for (int h=0;h<number_of_frames*N*(rad_buff+ang_buff);h++){
      descriptors[h]=0;
  }
  for (int h=0;h<number_of_frames*N*(rad_buff);h++){
      des3bsupp[h]=0;
  }
  for (int h=0;h<number_of_frames*N*(rad_buff*3);h++){
      der2b[h]=0;
  }
  for (int h=0;h<number_of_frames*N*(rad_buff*3);h++){
      der3bsupp[h]=0;
  }
  for (int h=0;h<number_of_frames*N*(ang_buff*2*3);h++){
      der3b[h]=0;
  }
  for (int h=0;h<number_of_frames*N;h++){
      numtriplet[h]=0;
  }



  for (int fr=0;fr<number_of_frames;fr++){

  for (int i=0;i<6;i++){
      box[i]=boxes[fr*6+i];
  }

  inobox[0]=1./box[0];
  inobox[1]=-box[1]/(box[0]*box[3]);
  inobox[2]=(box[1]*box[4])/(box[0]*box[3]*box[5])-box[2]/(box[0]*box[5]);
  inobox[3]=1./box[3];
  inobox[4]=-box[4]/(box[3]*box[5]);
  inobox[5]=1./box[5];


  for (int i=0;i<N;i++){
      float px=position[fr*N*3+i*3];
      float py=position[fr*N*3+i*3+1];
      float pz=position[fr*N*3+i*3+2];

      pos[i].x=(inobox[0]*px+inobox[1]*py+inobox[2]*pz);
      pos[i].y=(inobox[3]*py+inobox[4]*pz);
      pos[i].z=(inobox[5]*pz);

  }
  for (int i=0;i<N;i++)
  {
    ds_num[i]=0;
    ds_num_angular[i]=0;
    da_num[i]=0;
  }
  fullUpdateList(cells,pos,N,box,rc);
  resetInteractionMap(ime);
  calculateInteractionMapWithCutoffDistanceOrdered(cells,ime,pos,box,rc);
  buildImFromIme(ime,im);

  // I - PRIMI VICINI //////////////////////////////////////////////////////////

  for (int i=0;i<N;i++)
  {
    distsymm dij,buffer;
    int im_index=0;
    vector olddist,dist;
    int pos_index;

    while (im_index<im->howmany[i])
    {

      int j=im->with[i][im_index];
      dij.index=j;
      // calcolo per le derivate
      olddist.x=pos[i].x-pos[j].x;
      olddist.y=pos[i].y-pos[j].y;
      olddist.z=pos[i].z-pos[j].z;

      olddist.x-=rint(olddist.x);
      olddist.y-=rint(olddist.y);
      olddist.z-=rint(olddist.z);

      dist.x=box[0]*olddist.x+box[1]*olddist.y+box[2]*olddist.z;
      dist.y=box[3]*olddist.y+box[4]*olddist.z;
      dist.z=box[5]*olddist.z;

      dij.dist=sqrt(SQR(dist.x)+SQR(dist.y)+SQR(dist.z));

      dij.dx=dist.x;
      dij.dy=dist.y;
      dij.dz=dist.z;


      pos_index=insertionSort_distsymm(ds[i],&dij,&buffer,ds_num+i,sizeof(distsymm),compare_distsymm);

      if (ds_num[i]>n_input){
        printf("Buffer radiale saturato");
         exit(1);
       }

      dij.index=i;

      dij.dx*=-1;
      dij.dy*=-1;
      dij.dz*=-1;

      pos_index=insertionSort_distsymm(ds[j],&dij,&buffer,ds_num+j,sizeof(distsymm),compare_distsymm);

      if (ds_num[j]>n_input){
            printf("Buffer radiale saturato");
         exit(1);
       }
      if (dij.dist<range_angolare)
        {
          ds_num_angular[i]++;
          ds_num_angular[j]++;
        }


      im_index++;

    }

}


  //CON NINPUT PARI A
  float pi=3.1415926535;
  for (int i=0;i<N;i++)
  {
    int j,k;

    distangle da_ijk,buffer;
    vector olddist;

    for (j=0;j<ds_num_angular[i]-1;j++)
    {
      for (k=j+1;k<ds_num_angular[i];k++)
      {

        da_ijk.indexj=ds[i][j].index;
        da_ijk.indexk=ds[i][k].index;
        da_ijk.distj=ds[i][j].dist;
        da_ijk.distk=ds[i][k].dist;
        float angle=(ds[i][j].dx*ds[i][k].dx+ds[i][j].dy*ds[i][k].dy+ds[i][j].dz*ds[i][k].dz)/(da_ijk.distj*da_ijk.distk);

	//Here we implement a cosine cutoff
        float cutoffj,dcij;
        cutoffj=0.5*(1.+cos(pi*da_ijk.distj/range_angolare));
        dcij=-0.5*sin(pi*da_ijk.distj/range_angolare)*pi/range_angolare/da_ijk.distj;
        float cutoffk,dcik;
        cutoffk=0.5*(1.+cos(pi*da_ijk.distk/range_angolare));
        dcik=-0.5*sin(pi*da_ijk.distk/range_angolare)*pi/range_angolare/da_ijk.distk;
        da_ijk.angle=0.5*(angle+1)*cutoffj*cutoffk;


        float xij=ds[i][j].dx;
        float yij=ds[i][j].dy;
        float zij=ds[i][j].dz;
        float dij2=SQR(ds[i][j].dist);
        float xik=ds[i][k].dx;
        float yik=ds[i][k].dy;
        float zik=ds[i][k].dz;
        float dik2=SQR(ds[i][k].dist);

        float dcxij=dcij*ds[i][j].dx;
        float dcyij=dcij*ds[i][j].dy;
        float dczij=dcij*ds[i][j].dz;

        float dcxik=dcik*ds[i][k].dx;
        float dcyik=dcik*ds[i][k].dy;
        float dczik=dcik*ds[i][k].dz;

        float danglexij=0.5*(dij2*xik - xij*(xij*xik + yij*yik + zij*zik))/(Power(dij2,1.5)*Sqrt(dik2));
        float dangleyij=0.5*(dij2*yik - yij*(xij*xik + yij*yik + zij*zik))/(Power(dij2,1.5)*Sqrt(dik2));
        float danglezij=0.5*(dij2*zik - zij*(xij*xik + yij*yik + zij*zik))/(Power(dij2,1.5)*Sqrt(dik2));

        float danglexik=0.5*(dik2*xij - xik*(xij*xik + yij*yik + zij*zik))/(Sqrt(dij2)*Power(dik2,1.5));
        float dangleyik=0.5*(dik2*yij - yik*(xij*xik + yij*yik + zij*zik))/(Sqrt(dij2)*Power(dik2,1.5));
        float danglezik=0.5*(dik2*zij - zik*(xij*xik + yij*yik + zij*zik))/(Sqrt(dij2)*Power(dik2,1.5));

        da_ijk.dxij=danglexij*cutoffj*cutoffk+0.5*(angle+1)*dcxij*cutoffk;
        da_ijk.dyij=dangleyij*cutoffj*cutoffk+0.5*(angle+1)*dcyij*cutoffk;
        da_ijk.dzij=danglezij*cutoffj*cutoffk+0.5*(angle+1)*dczij*cutoffk;

        da_ijk.dxik=danglexik*cutoffj*cutoffk+0.5*(angle+1)*cutoffj*dcxik;
        da_ijk.dyik=dangleyik*cutoffj*cutoffk+0.5*(angle+1)*cutoffj*dcyik;
        da_ijk.dzik=danglezik*cutoffj*cutoffk+0.5*(angle+1)*cutoffj*dczik;


        int ins=insertionSort_distangle(da[i],&da_ijk,&buffer,da_num+i,sizeof(distangle),notcompare);


        if (da_num[i]>n_input_angolare)
          {
           printf("Buffer angolare saturato!");
           exit(1);
         }
      }
    }
 }
////OUTPUT///

   for (int i=0;i<N;i++){
        int actual_pos=fr*N*(radial_buffer+angular_buffer)+i*(radial_buffer+angular_buffer);
        intmap2b[fr*N*(radial_buffer+1)+i*(radial_buffer+1)]=ds_num[i];
        numtriplet[fr*N+i]=ds_num_angular[i];

    for (int k=0;k<ds_num_angular[i];k++){
        if (ds[i][k].dist<Rs){
          descriptors[actual_pos+k]=coeffA/Power(ds[i][k].dist,Pow_alpha)+coeffB/Power(ds[i][k].dist,Pow_beta)+coeffC;

          der2b[fr*N*3*radial_buffer+i*3*radial_buffer+k]=(-Pow_alpha*coeffA/Power(ds[i][k].dist,Pow_alpha+1.)-Pow_beta*coeffB/Power(ds[i][k].dist,Pow_beta+1.))*ds[i][k].dx/ds[i][k].dist;
          der2b[fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k]=(-Pow_alpha*coeffA/Power(ds[i][k].dist,Pow_alpha+1.)-Pow_beta*coeffB/Power(ds[i][k].dist,Pow_beta+1.))*ds[i][k].dy/ds[i][k].dist;
          der2b[fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k]=(-Pow_alpha*coeffA/Power(ds[i][k].dist,Pow_alpha+1.)-Pow_beta*coeffB/Power(ds[i][k].dist,Pow_beta+1.))*ds[i][k].dz/ds[i][k].dist;


        }
        else{
        descriptors[actual_pos+k]=0.5*(cos(pi*ds[i][k].dist/range)+1);
        der2b[fr*N*3*radial_buffer+i*3*radial_buffer+k]=-0.5*sin(pi*ds[i][k].dist/range)*pi/range*ds[i][k].dx/ds[i][k].dist;
        der2b[fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k]=-0.5*sin(pi*ds[i][k].dist/range)*pi/range*ds[i][k].dy/ds[i][k].dist;
        der2b[fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k]=-0.5*sin(pi*ds[i][k].dist/range)*pi/range*ds[i][k].dz/ds[i][k].dist;
      }
        des3bsupp[fr*N*radial_buffer+i*radial_buffer+k]=0.5*(cos(pi*ds[i][k].dist/range_angolare)+1);
        der3bsupp[fr*N*3*radial_buffer+i*3*radial_buffer+k]=-0.5*sin(pi*ds[i][k].dist/range_angolare)*pi/range_angolare*ds[i][k].dx/ds[i][k].dist;
        der3bsupp[fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k]=-0.5*sin(pi*ds[i][k].dist/range_angolare)*pi/range_angolare*ds[i][k].dy/ds[i][k].dist;
        der3bsupp[fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k]=-0.5*sin(pi*ds[i][k].dist/range_angolare)*pi/range_angolare*ds[i][k].dz/ds[i][k].dist;
        intmap2b[fr*N*(radial_buffer+1)+i*(radial_buffer+1)+1+k]=ds[i][k].index;

      }
      for (int k=ds_num_angular[i];k<ds_num[i];k++){
        intmap2b[fr*N*(radial_buffer+1)+i*(radial_buffer+1)+1+k]=ds[i][k].index;
        if (ds[i][k].dist<Rs){
          descriptors[actual_pos+k]=coeffA/Power(ds[i][k].dist,Pow_alpha)+coeffB/Power(ds[i][k].dist,Pow_beta)+coeffC;

          der2b[fr*N*3*radial_buffer+i*3*radial_buffer+k]=(-Pow_alpha*coeffA/Power(ds[i][k].dist,Pow_alpha+1)-Pow_beta*coeffB/Power(ds[i][k].dist,Pow_beta+1))*ds[i][k].dx/ds[i][k].dist;
          der2b[fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k]=(-Pow_alpha*coeffA/Power(ds[i][k].dist,Pow_alpha+1)-Pow_beta*coeffB/Power(ds[i][k].dist,Pow_beta+1))*ds[i][k].dy/ds[i][k].dist;
          der2b[fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k]=(-Pow_alpha*coeffA/Power(ds[i][k].dist,Pow_alpha+1)-Pow_beta*coeffB/Power(ds[i][k].dist,Pow_beta+1))*ds[i][k].dz/ds[i][k].dist;

        }
        else{
        descriptors[actual_pos+k]=0.5*(cos(pi*ds[i][k].dist/range)+1);

        der2b[fr*N*3*radial_buffer+i*3*radial_buffer+k]=-0.5*sin(pi*ds[i][k].dist/range)*pi/range*ds[i][k].dx/ds[i][k].dist;
        der2b[fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k]=-0.5*sin(pi*ds[i][k].dist/range)*pi/range*ds[i][k].dy/ds[i][k].dist;
        der2b[fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k]=-0.5*sin(pi*ds[i][k].dist/range)*pi/range*ds[i][k].dz/ds[i][k].dist;
      }
      }

    int ang_neigh=ds_num_angular[i];
    int na_real=ang_neigh*(ang_neigh-1)/2;
    for (int k=0;k<na_real;k++){
        descriptors[actual_pos+radial_buffer+k]=da[i][k].angle;
        intmap3b[fr*N*(ang_buff*2)+i*(ang_buff*2)+k*2]=da[i][k].indexj;
        intmap3b[fr*N*(ang_buff*2)+i*(ang_buff*2)+k*2+1]=da[i][k].indexk;

        der3b[fr*N*3*2*ang_buff+i*(3*ang_buff*2)+2*k]=da[i][k].dxij;
        der3b[fr*N*3*2*ang_buff+i*(3*ang_buff*2)+2*k+1]=da[i][k].dxik;

        der3b[fr*N*3*2*ang_buff+i*(3*ang_buff*2)+2*ang_buff+2*k]=da[i][k].dyij;
        der3b[fr*N*3*2*ang_buff+i*(3*ang_buff*2)+2*ang_buff+2*k+1]=da[i][k].dyik;

        der3b[fr*N*3*2*ang_buff+i*(3*ang_buff*2)+2*ang_buff*2+2*k]=da[i][k].dzij;
        der3b[fr*N*3*2*ang_buff+i*(3*ang_buff*2)+2*ang_buff*2+2*k+1]=da[i][k].dzik;
        }

  }
}
   //Qui andrebbero riazzrati tutti o liberati, ds e da andrebbero riazzerati ma
   ////se tutto funziona non serve
}

int main(int argv,char* argc){

    float R_c=4.5;
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
    read_data_float("serial_benchmark/input_data/box_4debug_flat",nowbox);
    read_data_float("serial_benchmark/input_data/pos_4debug_flat",nowpos);

    int *howmany_d;
    int *with_d;
    cudaMalloc(&howmany_d,nf*N*sizeof(int));
    cudaMalloc(&with_d,nf*N*radbuff*sizeof(int));


    //Construct descriptor utilities
    construct_descriptor(nowbox,R_c,N,
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
        float px=nowpos[fr*N*3+i*3];
        float py=nowpos[fr*N*3+i*3+1];
        float pz=nowpos[fr*N*3+i*3+2];

        pos[i].x=(inobox[ii*6+0]*px+inobox[ii*6+1]*py+inobox[ii*6+2]*pz);
        pos[i].y=(inobox[ii*6+3]*py+inobox[ii*6+4]*pz);
        pos[i].z=(inobox[ii*6+5]*pz);

        nowpos[fr*N*3+i*3]=pos[i].x;
        nowpos[fr*N*3+i*3+1]=pos[i].y;
        nowpos[fr*N*3+i*3+2]=pos[i].z;
      }

      // calcolo delle celle e dei neighbour list
      fullUpdateList(cells,pos,N,box,rc);
      resetInteractionMap(ime);
      calculateInteractionMapWithCutoffDistanceOrdered(cells,ime,pos,box,rc);
      //buildImFromIme(ime,im);
      im=ime;



      cudaMemcpy(howmany_d+ii*N,im->howmany,N*sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(with_d+ii*N*radbuff,im->with,N*radbuff*sizeof(int),cudaMemcpyHostToDevice);

      for (i=0;i<N;i++)
      {
        if (im->howmany[i]>n_input)
        {
          printf("Buffer radiale saturato\n");
          exit(1);
        }

    }



    float *box_d;
    float *inobox_d;
    cudaMalloc(&box_d,nf*6*sizeof(float));
    cudaMalloc(&inobox_d,nf*6*sizeof(float));

    cudaMemcpy(box_d,nowbox,nf*6*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(inobox_d,nowinobox,nf*6*sizeof(float),cudaMemcpyHostToDevice);



    //alloco array per outputs
    /* float*  descriptor_tensor=(float*)calloc(b*N*(radbuff+angbuff),sizeof(float));
    int* intmap2b_tensor=(int*)calloc(b*N*(radbuff+1),sizeof(int));
    int* intmap3b_tensor=(int*)calloc(b*N*(angbuff*2),sizeof(int));
    float* des3bsupp_tensor=(float*)calloc(b*N*(radbuff),sizeof(float));
    float* der3b_tensor=(float*)calloc(b*N*(angbuff*2*3),sizeof(float));
    float* der2b_tensor=(float*)calloc(b*N*(radbuff*3),sizeof(float));
    float* der3bsupp_tensor=(float*)calloc(b*N*(radbuff*3),sizeof(float));
    int* numtriplet_tensor=  (int*)calloc(b*N,sizeof(int)); */




    float* descriptor_tensor;
    int* intmap2b_tensor;
    int* intmap3b_tensor;
    float* des3bsupp_tensor;
    float* der3b_tensor;
    float* der2b_tensor;
    float* der3bsupp_tensor;
    int* numtriplet_tensor;

    // Allocate device memory
    cudaMalloc((void**)&descriptor_tensor, b * N * (radbuff + angbuff) * sizeof(float));
    cudaMalloc((void**)&intmap2b_tensor, b * N * (radbuff + 1) * sizeof(int));
    cudaMalloc((void**)&intmap3b_tensor, b * N * (angbuff * 2) * sizeof(int));
    cudaMalloc((void**)&des3bsupp_tensor, b * N * (radbuff) * sizeof(float));
    cudaMalloc((void**)&der3b_tensor, b * N * (angbuff * 2 * 3) * sizeof(float));
    cudaMalloc((void**)&der2b_tensor, b * N * (radbuff * 3) * sizeof(float));
    cudaMalloc((void**)&der3bsupp_tensor, b * N * (radbuff * 3) * sizeof(float));
    cudaMalloc((void**)&numtriplet_tensor, b * N * sizeof(int));

    // Initialize allocated memory to zero
    cudaMemset(descriptor_tensor, 0, b * N * (radbuff + angbuff) * sizeof(float));
    cudaMemset(intmap2b_tensor, 0, b * N * (radbuff + 1) * sizeof(int));
    cudaMemset(intmap3b_tensor, 0, b * N * (angbuff * 2) * sizeof(int));
    cudaMemset(des3bsupp_tensor, 0, b * N * (radbuff) * sizeof(float));
    cudaMemset(der3b_tensor, 0, b * N * (angbuff * 2 * 3) * sizeof(float));
    cudaMemset(der2b_tensor, 0, b * N * (radbuff * 3) * sizeof(float));
    cudaMemset(der3bsupp_tensor, 0, b * N * (radbuff * 3) * sizeof(float));
    cudaMemset(numtriplet_tensor, 0, b * N * sizeof(int));


    /* for (int h=0;h<number_of_frames*N*(rad_buff+1);h++){
        intmap2b[h]=0;
    }

    for (int h=0;h<number_of_frames*N*(ang_buff*2);h++){
        intmap3b[h]=0;
    }
    for (int h=0;h<number_of_frames*N*(rad_buff+ang_buff);h++){
        descriptors[h]=0;
    }
    for (int h=0;h<number_of_frames*N*(rad_buff);h++){
        des3bsupp[h]=0;
    }
    for (int h=0;h<number_of_frames*N*(rad_buff*3);h++){
        der2b[h]=0;
    }
    for (int h=0;h<number_of_frames*N*(rad_buff*3);h++){
        der3bsupp[h]=0;
    }
    for (int h=0;h<number_of_frames*N*(ang_buff*2*3);h++){
        der3b[h]=0;
    }
    for (int h=0;h<number_of_frames*N;h++){
        numtriplet[h]=0;
    } */







    //compute descriptors
    computeDescriptors(R_c,radbuff,R_a,angbuff,N,
                          nowpos,box_d,inobox_d,
                          howmany_d,with_d,
                          descriptor_tensor,intmap2b_tensor,intmap3b_tensor,
                          des3bsupp_tensor,der3b_tensor,der2b_tensor,
                          der3bsupp_tensor,nf,numtriplet_tensor);
    int k;
    for (k=0;k<b*N*angbuff*2*3;k++){
    printf("%g\n",der3b_tensor[k]);
    }

}
