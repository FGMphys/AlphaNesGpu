#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <ctype.h>


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"



#include "vector.h"
#include "interaction_map.h"
#include "cell_list.h"
#include "smart_allocator.h"
#include "utilities.h"
#include "log.h"


#define SQR(x) ((x)*(x))
#define Sqrt(x) (sqrt(x))
#define Power(x,n) (pow(x,n))

#define MAX_LINE_LENGTH 2000
#define MAX_NNEIGHBOURS 300

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


using namespace tensorflow;

REGISTER_OP("ConstructDescriptorsLight")
    .Input("radial_cutoff: float")
    .Input("radial_buffer: int32")
    .Input("angular_buffer: int32")
    .Input("numpar: int32")
    .Input("boxer: float")
    .Input("rs: float")
    .Output("exitcode: int32");

    void construct_descriptor(const Tensor& box_T,const Tensor& rc_T,const Tensor& N_T,
                              const Tensor& rad_buffer_T,const Tensor& angular_buffer_T){
          // GENERATE CELLS
    //      auto position=position_T.flat<float>();
          auto box_flat=box_T.flat<float>();
          for (int i=0;i<6;i++){
              box[i]=box_flat(i);
          }
          inobox[0]=1./box[0];
          inobox[1]=-box[1]/(box[0]*box[3]);
          inobox[2]=(box[1]*box[4])/(box[0]*box[3]*box[5])-box[2]/(box[0]*box[5]);
          inobox[3]=1./box[3];
          inobox[4]=-box[4]/(box[3]*box[5]);
          inobox[5]=1./box[5];

          auto rc_flat=rc_T.flat<float>();
          auto N_flat=N_T.flat<int>();

          float rc=rc_flat(0);
          N=N_flat(0);



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

          auto rad_buffer_flat=rad_buffer_T.flat<int>();
          auto angular_buffer_flat=angular_buffer_T.flat<int>();

          n_input=rad_buffer_flat(0);
          n_input_angolare=angular_buffer_flat(0);
          for (int i=0;i<N;i++)
          {
            ds[i]=(distsymm*)calloc(n_input,sizeof(distsymm));
            da[i]=(distangle*)calloc(n_input_angolare,sizeof(distangle));
            ds_num[i]=0;
            ds_num_angular[i]=0;
            da_num[i]=0;
          }

    }

    class ConstructDescriptorsLightOp : public OpKernel {
     public:
      explicit ConstructDescriptorsLightOp(OpKernelConstruction* context) : OpKernel(context) {


      }

      void Compute(OpKernelContext* context) override {
           const Tensor& rcrad_T = context->input(0);
           const Tensor& radbuff_T = context->input(1);
           const Tensor& angbuff_T = context->input(2);
           const Tensor& numpar_T = context->input(3);
           const Tensor& box_T = context->input(4);

           //Metto fisso Rs
           const Tensor& rs_T = context->input(5);
           auto rs_T_flat=rs_T.flat<float>();
           Rs=rs_T_flat(0);
           auto rcrad_T_flat=rcrad_T.flat<float>();
           float rc=rcrad_T_flat(0);
           construct_repulsion(rc);
           construct_descriptor(box_T,rcrad_T,numpar_T,radbuff_T,angbuff_T);
       }
       };
       REGISTER_KERNEL_BUILDER(Name("ConstructDescriptorsLight").Device(DEVICE_CPU), ConstructDescriptorsLightOp);






void make_descriptors(float rc,int rad_buff,float rc_ang,int ang_buff,int N,
                      const Tensor& position_T,const Tensor& box_T,
                      Tensor* descriptors_T,Tensor* intmap2b_T,Tensor* intmap3b_T,
                      Tensor* des3bsupp_T,Tensor* der3b_T,Tensor* der2b_T,
                      Tensor* der3bsupp_T, int number_of_frames,Tensor* numtriplet_T)
{


  float range=rc;
  float range_angolare=rc_ang;
  int radial_buffer=rad_buff;
  int angular_buffer=ang_buff;
  auto boxer=box_T.flat<float>();
  auto position=position_T.flat<float>();
  //Output Tensor
  auto intmap2b=intmap2b_T->flat<int>();
  auto intmap3b=intmap3b_T->flat<int>();
  auto descriptors=descriptors_T->flat<float>();
  auto des3bsupp=des3bsupp_T->flat<float>();
  auto der2b=der2b_T->flat<float>();
  auto der3bsupp=der3bsupp_T->flat<float>();
  auto der3b=der3b_T->flat<float>();
  auto numtriplet=numtriplet_T->flat<int>();

  for (int h=0;h<number_of_frames*N*(rad_buff+1);h++){
      intmap2b(h)=0;
  }

  for (int h=0;h<number_of_frames*N*(ang_buff*2);h++){
      intmap3b(h)=0;
  }
  for (int h=0;h<number_of_frames*N*(rad_buff+ang_buff);h++){
      descriptors(h)=0;
  }
  for (int h=0;h<number_of_frames*N*(rad_buff);h++){
      des3bsupp(h)=0;
  }
  for (int h=0;h<number_of_frames*N*(rad_buff*3);h++){
      der2b(h)=0;
  }
  for (int h=0;h<number_of_frames*N*(rad_buff*3);h++){
      der3bsupp(h)=0;
  }
  for (int h=0;h<number_of_frames*N*(ang_buff*2*3);h++){
      der3b(h)=0;
  }
  for (int h=0;h<number_of_frames*N;h++){
      numtriplet(h)=0;
  }



  for (int fr=0;fr<number_of_frames;fr++){

  for (int i=0;i<6;i++){
      box[i]=boxer(fr*6+i);
  }
  inobox[0]=1./box[0];
  inobox[1]=-box[1]/(box[0]*box[3]);
  inobox[2]=(box[1]*box[4])/(box[0]*box[3]*box[5])-box[2]/(box[0]*box[5]);
  inobox[3]=1./box[3];
  inobox[4]=-box[4]/(box[3]*box[5]);
  inobox[5]=1./box[5];

///////////////work
//

  for (int i=0;i<N;i++){
      float px=position(fr*N*3+i*3);
      float py=position(fr*N*3+i*3+1);
      float pz=position(fr*N*3+i*3+2);

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
        intmap2b(fr*N*(radial_buffer+1)+i*(radial_buffer+1))=ds_num[i];
        numtriplet(fr*N+i)=ds_num_angular[i];

    for (int k=0;k<ds_num_angular[i];k++){
        if (ds[i][k].dist<Rs){
          descriptors(actual_pos+k)=coeffA/Power(ds[i][k].dist,Pow_alpha)+coeffB/Power(ds[i][k].dist,Pow_beta)+coeffC;

          der2b(fr*N*3*radial_buffer+i*3*radial_buffer+k)=(-Pow_alpha*coeffA/Power(ds[i][k].dist,Pow_alpha+1.)-Pow_beta*coeffB/Power(ds[i][k].dist,Pow_beta+1.))*ds[i][k].dx/ds[i][k].dist;
          der2b(fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k)=(-Pow_alpha*coeffA/Power(ds[i][k].dist,Pow_alpha+1.)-Pow_beta*coeffB/Power(ds[i][k].dist,Pow_beta+1.))*ds[i][k].dy/ds[i][k].dist;
          der2b(fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k)=(-Pow_alpha*coeffA/Power(ds[i][k].dist,Pow_alpha+1.)-Pow_beta*coeffB/Power(ds[i][k].dist,Pow_beta+1.))*ds[i][k].dz/ds[i][k].dist;


        }
        else{
        descriptors(actual_pos+k)=0.5*(cos(pi*ds[i][k].dist/range)+1);
        der2b(fr*N*3*radial_buffer+i*3*radial_buffer+k)=-0.5*sin(pi*ds[i][k].dist/range)*pi/range*ds[i][k].dx/ds[i][k].dist;
        der2b(fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k)=-0.5*sin(pi*ds[i][k].dist/range)*pi/range*ds[i][k].dy/ds[i][k].dist;
        der2b(fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k)=-0.5*sin(pi*ds[i][k].dist/range)*pi/range*ds[i][k].dz/ds[i][k].dist;
      }
        des3bsupp(fr*N*radial_buffer+i*radial_buffer+k)=0.5*(cos(pi*ds[i][k].dist/range_angolare)+1);
        der3bsupp(fr*N*3*radial_buffer+i*3*radial_buffer+k)=-0.5*sin(pi*ds[i][k].dist/range_angolare)*pi/range_angolare*ds[i][k].dx/ds[i][k].dist;
        der3bsupp(fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k)=-0.5*sin(pi*ds[i][k].dist/range_angolare)*pi/range_angolare*ds[i][k].dy/ds[i][k].dist;
        der3bsupp(fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k)=-0.5*sin(pi*ds[i][k].dist/range_angolare)*pi/range_angolare*ds[i][k].dz/ds[i][k].dist;
        intmap2b(fr*N*(radial_buffer+1)+i*(radial_buffer+1)+1+k)=ds[i][k].index;

      }
      for (int k=ds_num_angular[i];k<ds_num[i];k++){
        intmap2b(fr*N*(radial_buffer+1)+i*(radial_buffer+1)+1+k)=ds[i][k].index;
        if (ds[i][k].dist<Rs){
          descriptors(actual_pos+k)=coeffA/Power(ds[i][k].dist,Pow_alpha)+coeffB/Power(ds[i][k].dist,Pow_beta)+coeffC;;

          der2b(fr*N*3*radial_buffer+i*3*radial_buffer+k)=(-Pow_alpha*coeffA/Power(ds[i][k].dist,Pow_alpha+1)-Pow_beta*coeffB/Power(ds[i][k].dist,Pow_beta+1))*ds[i][k].dx/ds[i][k].dist;;
          der2b(fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k)=(-Pow_alpha*coeffA/Power(ds[i][k].dist,Pow_alpha+1)-Pow_beta*coeffB/Power(ds[i][k].dist,Pow_beta+1))*ds[i][k].dy/ds[i][k].dist;;
          der2b(fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k)=(-Pow_alpha*coeffA/Power(ds[i][k].dist,Pow_alpha+1)-Pow_beta*coeffB/Power(ds[i][k].dist,Pow_beta+1))*ds[i][k].dz/ds[i][k].dist;;

        }
        else{
        descriptors(actual_pos+k)=0.5*(cos(pi*ds[i][k].dist/range)+1);

        der2b(fr*N*3*radial_buffer+i*3*radial_buffer+k)=-0.5*sin(pi*ds[i][k].dist/range)*pi/range*ds[i][k].dx/ds[i][k].dist;
        der2b(fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer+k)=-0.5*sin(pi*ds[i][k].dist/range)*pi/range*ds[i][k].dy/ds[i][k].dist;
        der2b(fr*N*3*radial_buffer+i*3*radial_buffer+radial_buffer*2+k)=-0.5*sin(pi*ds[i][k].dist/range)*pi/range*ds[i][k].dz/ds[i][k].dist;
      }
      }

    int ang_neigh=ds_num_angular[i];
    int na_real=ang_neigh*(ang_neigh-1)/2;
    for (int k=0;k<na_real;k++){
        descriptors(actual_pos+radial_buffer+k)=da[i][k].angle;
        intmap3b(fr*N*(ang_buff*2)+i*(ang_buff*2)+k*2)=da[i][k].indexj;
        intmap3b(fr*N*(ang_buff*2)+i*(ang_buff*2)+k*2+1)=da[i][k].indexk;

        der3b(fr*N*3*2*ang_buff+i*(3*ang_buff*2)+2*k)=da[i][k].dxij;
        der3b(fr*N*3*2*ang_buff+i*(3*ang_buff*2)+2*k+1)=da[i][k].dxik;

        der3b(fr*N*3*2*ang_buff+i*(3*ang_buff*2)+2*ang_buff+2*k)=da[i][k].dyij;
        der3b(fr*N*3*2*ang_buff+i*(3*ang_buff*2)+2*ang_buff+2*k+1)=da[i][k].dyik;

        der3b(fr*N*3*2*ang_buff+i*(3*ang_buff*2)+2*ang_buff*2+2*k)=da[i][k].dzij;
        der3b(fr*N*3*2*ang_buff+i*(3*ang_buff*2)+2*ang_buff*2+2*k+1)=da[i][k].dzik;
        }

  }
}
   //Qui andrebbero riazzrati tutti o liberati, ds e da andrebbero riazzerati ma
   ////se tutto funziona non serve
}



REGISTER_OP("ComputeDescriptorsLight")
    .Input("radial_cutoff: float")
    .Input("radial_buffer: int32")
    .Input("angular_cutoff: float")
    .Input("angular_buffer: int32")
    .Input("numpar: int32")
    .Input("positions: float")
    .Input("boxer: float")
    .Input("number_of_frames: int32")
    .Output("descriptors: float")
    .Output("des3bsupp: float")
    .Output("intmap2b: int32")
    .Output("intmap3b: int32")
    .Output("der2b: float")
    .Output("der3b: float")
    .Output("der3bsupp: float")
    .Output("numtriplet: int32");


class ComputeDescriptorsLightOp : public OpKernel {
 public:
  explicit ComputeDescriptorsLightOp(OpKernelConstruction* context) : OpKernel(context) {


  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& rcrad_T = context->input(0);
    const Tensor& radbuff_T = context->input(1);
    const Tensor& rcang_T = context->input(2);
    const Tensor& angbuff_T = context->input(3);
    const Tensor& numpar_T = context->input(4);
    const Tensor& positions_T = context->input(5);
    const Tensor& box_T = context->input(6);
    const Tensor& numframes_T = context->input(7);


    auto rcrad_T_flat = rcrad_T.flat<float>();
    auto radbuff_T_flat = radbuff_T.flat<int>();
    auto rcang_T_flat = rcang_T.flat<float>();
    auto angbuff_T_flat = angbuff_T.flat<int>();
    auto numpar_T_flat = numpar_T.flat<int>();
    auto numframes_T_flat=numframes_T.flat<int>();


    //Copio i tensori input in nuovi array per elaborarli
    float rc=rcrad_T_flat(0);
    int radial_buffer=radbuff_T_flat(0);
    float rc_ang=rcang_T_flat(0);
    int ang_buffer=angbuff_T_flat(0);
    int N=numpar_T_flat(0);
    int nf=numframes_T_flat(0);



    ///////////////DESCRIPTORS///////////////
    // Create an output tensor
    Tensor* descriptor_tensor = NULL;
    TensorShape descriptor_shape;
    descriptor_shape.AddDim (nf);
    descriptor_shape.AddDim (N);
    descriptor_shape.AddDim (radial_buffer+ang_buffer);
    OP_REQUIRES_OK(context, context->allocate_output(0,descriptor_shape,
                                                     &descriptor_tensor));
    ///////////////DESCRIPTORS 3B SUPP///////////////
    // Create an output tensor
    Tensor* des3bsupp_tensor = NULL;
    TensorShape des3bsupp_shape;
    des3bsupp_shape.AddDim (nf);
    des3bsupp_shape.AddDim (N);
    des3bsupp_shape.AddDim (radial_buffer);
    OP_REQUIRES_OK(context, context->allocate_output(1,des3bsupp_shape,
                                                     &des3bsupp_tensor));
    ///////////////INTMAP2B///////////////
    // Create an output tensor
    Tensor* intmap2b_tensor = NULL;
    TensorShape intmap2b_shape;
    intmap2b_shape.AddDim (nf);
    intmap2b_shape.AddDim (N);
    intmap2b_shape.AddDim (radial_buffer+1);
    OP_REQUIRES_OK(context, context->allocate_output(2,intmap2b_shape,
                                                     &intmap2b_tensor));
    /////////////////////////////
    ///////////////INTMAP3B///////////////
    // Create an output tensor
    Tensor* intmap3b_tensor = NULL;
    TensorShape intmap3b_shape;
    intmap3b_shape.AddDim (nf);
    intmap3b_shape.AddDim (N);
    intmap3b_shape.AddDim (ang_buffer*2);
    OP_REQUIRES_OK(context, context->allocate_output(3,intmap3b_shape,
                                                     &intmap3b_tensor));
    /////////////////////////////
    ///////////////DER2B///////////////
    // Create an output tensor
    Tensor* der2b_tensor = NULL;
    TensorShape der2b_shape;
    der2b_shape.AddDim (nf);
    der2b_shape.AddDim (N);
    der2b_shape.AddDim (3);
    der2b_shape.AddDim (radial_buffer);
    OP_REQUIRES_OK(context, context->allocate_output(4,der2b_shape,
                                                     &der2b_tensor));
    /////////////////////////////
    ///////////////DER3B///////////////
    // Create an output tensor
    Tensor* der3b_tensor = NULL;
    TensorShape der3b_shape;
    der3b_shape.AddDim (nf);
    der3b_shape.AddDim (N);
    der3b_shape.AddDim (3);
    der3b_shape.AddDim (ang_buffer*2);
    OP_REQUIRES_OK(context, context->allocate_output(5,der3b_shape,
                                                     &der3b_tensor));


    /////////////////////////////
    ///////////////DER3B_SUPP///////////////
    // Create an output tensor
    Tensor* der3bsupp_tensor = NULL;
    TensorShape der3bsupp_shape;
    der3bsupp_shape.AddDim (nf);
    der3bsupp_shape.AddDim (N);
    der3bsupp_shape.AddDim (3);
    der3bsupp_shape.AddDim (radial_buffer);
    OP_REQUIRES_OK(context, context->allocate_output(6,der3bsupp_shape,
                                                     &der3bsupp_tensor));

    // Create an output tensor
    Tensor* numtriplet_tensor = NULL;
    TensorShape numtriplet_shape;
    numtriplet_shape.AddDim (nf);
    numtriplet_shape.AddDim (N);
    OP_REQUIRES_OK(context, context->allocate_output(7,numtriplet_shape,
                                                     &numtriplet_tensor));
    /////////////////////////////
    make_descriptors(rc,radial_buffer,rc_ang,ang_buffer,N,
                          positions_T,box_T,
                          descriptor_tensor,intmap2b_tensor,intmap3b_tensor,
                          des3bsupp_tensor,der3b_tensor,der2b_tensor,
                          der3bsupp_tensor,nf,numtriplet_tensor);
     }




   };
   REGISTER_KERNEL_BUILDER(Name("ComputeDescriptorsLight").Device(DEVICE_CPU), ComputeDescriptorsLightOp);
