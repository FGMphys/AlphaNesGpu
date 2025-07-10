#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "vector.h"
#include "global_definitions.h"
#include "log.h"
#include "smart_allocator.h"
#include "io.h"


#define SQR(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))

#define NMAX 100000
typedef struct _grstruct {
	int nsamples;       // number of samples added to histogram
	double bin;         // bin size
	double *histogram;  // histogram
	double *volume_shell;
	int dimh;           // dimensions of the histogram
	double range;       // bin*dimh
	double Zns_over_Zs;
} grstruct;


grstruct* grConstruct(double binsize,double range);


void grSample(grstruct *gr,vector *pos,int numparticles,double box[],int* type_map,int nt,int nt2,int* type,double delta_E);
void grAverage(grstruct *gr,int numparticles,double box[]);

void grFree(grstruct *gr);

void grPrint(grstruct *gr,FILE *p_file);
void grSave(grstruct *gr,FILE *p_file);
int found_vec_dim(char* root_path){
	  char line[MAX_LINE_LENGTH]="";
    char filename[100];

    sprintf(filename,"%s",root_path);
    fflush(stdout);
    FILE *pfile=fopen(filename,"r");
    int dim=0;
    while (getLine(line,pfile)!=0)
    {
		dim++;
    }

    fclose(pfile);
    return dim;
}

void read_float_vec(double* vec,int dim,char* root_path)
{
        char line[MAX_LINE_LENGTH]="";
        char filename[100];
        sprintf(filename,"%s",root_path);
        fflush(stdout);
        FILE *pfile=fopen(filename,"r");
        int i=0;
        while (getLine(line,pfile)!=0)
        {
        sscanf(line,"%lf",vec+i);
        i++;
        }
  		  fclose(pfile);
}


void readType(int *Type_map,int N)
{
  FILE *pfile=fopen("type_map.dat","r");
  if (pfile==NULL)
     {
             printf("\nNo type_map.dat found!\n");
             exit(0);

     }
  int n=0;

  while(fscanf(pfile,"%d \n",Type_map+n)!=EOF)
  {
    n++;
  }


  fclose(pfile);

}

void read_tipos(int* tipos,int* numtypes)
{
        char line[MAX_LINE_LENGTH]="";
        FILE *pfile=fopen("type.dat","r");
        if (pfile==NULL){
             printf("Type.dat file noto found!");
             fflush(stdout);
		exit(0);
       }
        int i=0;
        while (getLine(line,pfile)!=0)
        {
	sscanf(line,"%d",tipos+i);
        printf("\n%d %d\n",i,tipos[i]);
	i++;
        }
	numtypes[0]=i;
  fclose(pfile);
}
int main(int argc,char *argv[])
{

	if (argc<3)
	{
		printf("%s [bin size] [delta_E*beta(non sampled-sampled)] [configuration file/s] \n",argv[0]);
		exit(1);
	}

	double bin_size=atof(argv[1]);
	char *delta_E_file=argv[2];

	int* type_map=calloc(NMAX,sizeof(int));
	int* type=calloc(100,sizeof(int));
	int numtypes[1];

	char **configuration_files=argv+3;
	int num_files=argc-3;

	//int dim=found_vec_dim(delta_E_file);

        double* delta_E_arr=(double*)calloc(num_files,sizeof(double));
	read_float_vec(delta_E_arr,num_files,delta_E_file);

	logStartStdout();


	//////////////////////////////////////////////////////////
	double min_box_size=10000000.;
	// dobbiamo costruire la struttura per il calcolo del gr
	// potenzialmente le configurazioni vengono da simulazioni NpT
	// in cui il lato della box cambia. Fissiamo come range la meta'
	// del lato piu' piccolo
	double box[6],ibox[6];
	steps step;
	int numparticles,old_numparticles;
	int i;
	int first_file=1;


        read_tipos(type,numtypes);
	int Nfromtipos=0;
	for (int k=0;k<numtypes[0];k++)
            Nfromtipos+=type[k];

        printf("\nNumber particles, %d number types %d!\n",Nfromtipos,numtypes[0]);
        fflush(stdout);
	readType(type_map,Nfromtipos);

	old_numparticles=0;
	double average_density=0.;

	for (i=0;i<num_files;i++)
	{
		getHeader(configuration_files[i],&step,&numparticles,box);


		if ((numparticles!=old_numparticles) && (first_file==0))
		{
			logPrint("Configurations have different number of particles\n");
			exit(1);
		}

		average_density+=(double)numparticles/(box[0]*box[3]*box[5]);

		if (box[0]<min_box_size)
			min_box_size=box[0];
		if (box[3]<min_box_size)
			min_box_size=box[3];
		if (box[5]<min_box_size)
			min_box_size=box[5];


		first_file=0;
		old_numparticles=numparticles;
	}

	average_density/=num_files;
	///////////////////////////////////////////////////////////////


	// costruiamo gr

	vector *pos=calloc(numparticles,sizeof(vector));

	for (int nt=0; nt<numtypes[0];nt++){
		for (int nt2=nt; nt2<numtypes[0];nt2++){
		    printf("Doing gr type %d %d!",nt,nt2);
		grstruct *gr=grConstruct(bin_size,0.5*min_box_size);
		for (i=0;i<num_files;i++)
			{
			readPositions(configuration_files[i],pos,&step,&numparticles,box,ibox);
	                double delta_E=delta_E_arr[i];
			grSample(gr,pos,numparticles,box,type_map,nt,nt2,type,delta_E);

			}


	char filename[100];
        sprintf(filename,"gr_%d_%d_rewe.dat",nt,nt2);
	FILE *pfile=fopen(filename,"w");

	printf("\nAveraging\n");
	grAverage(gr,type[nt2],box);
        printf("Saving\n");
        grSave(gr,pfile);

	grFree(gr);
	fclose(pfile);
	}
        }
	logClose();

	return 0;
}

grstruct* grConstruct(double binsize,double range)
{
	grstruct* gr=(grstruct*)malloc(sizeof(grstruct));

	gr->Zns_over_Zs=0.;

	gr->nsamples=0;

	gr->bin=binsize;

	gr->dimh=(int)(range/binsize);

	gr->range=(gr->dimh)*binsize;

	gr->histogram=(double*)calloc(gr->dimh,sizeof(double));

	gr->volume_shell=(double*)calloc(gr->dimh,sizeof(double));

	int i;
	for (i=0;i<(gr->dimh);i++)
	{
		gr->volume_shell[i]=(4./3.)*M_PI*(CUBE(i+1)-CUBE(i))*CUBE(gr->bin);
	}

	return gr;
}


void grSample(grstruct *gr,vector *pos,int numparticles,double Box[],int* type_map,int nt,int nt2,int* type,double delta_E)
{

	gr->nsamples++;

	int i;
	int j;
	vector olddist,dist;
	double dist2;
	int hpos;

	// attenzione la densita' potrebbe cambiare tra i diversi file
	// e quindi la campioniamo direttamente
	double two_over_density=2.*(Box[0]*Box[3]*Box[5])/(double)type[nt2];
	for (i=0;i<(numparticles-1);i++){
	    if (type_map[i]==nt){

		for (j=i+1;j<numparticles;j++)
		{
		        if (type_map[j]==nt2){
			olddist.x=pos[i].x-pos[j].x;
			olddist.y=pos[i].y-pos[j].y;
			olddist.z=pos[i].z-pos[j].z;

			olddist.x-=rint(olddist.x);
			olddist.y-=rint(olddist.y);
			olddist.z-=rint(olddist.z);

			dist.x=Box[0]*olddist.x+Box[1]*olddist.y+Box[2]*olddist.z;
			dist.y=Box[3]*olddist.y+Box[4]*olddist.z;
			dist.z=Box[5]*olddist.z;

			dist2=sqrt(SQR(dist.x)+SQR(dist.y)+SQR(dist.z));

			if (dist2<(gr->range))
			{

				hpos=(int)(dist2/gr->bin);

				gr->histogram[hpos]+=two_over_density*exp(-delta_E);
			}
		}
		}
	}
    }
}

void grAverage(grstruct *gr,int numparticles,double box[])
{
	int i;

	for (i=0;i<(gr->dimh);i++)
	{
		(gr->histogram[i])=(gr->histogram[i])/( (gr->nsamples)*numparticles*gr->volume_shell[i]);
	}
        //Normalizzo imponendo che l'ultimo r sia tale che g(r)=1
        /*for (i=0;i<(gr->dimh);i++)
        {
                (gr->histogram[i])=(gr->histogram[i])/gr->histogram[gr->dimh-1];
        }
        */
        //normalizzo imponendo che le ultime oscillazioni siano attorno a uno
        int frac=rint(0.2*gr->dimh);
        double mean_last_frac=0.;
        for (i=gr->dimh-frac;i<gr->dimh;i++)
        {
                mean_last_frac+=(gr->histogram[i])/frac;
        }
	printf("norm fact %lf\n",mean_last_frac);
        for (i=0;i<(gr->dimh);i++)
        {
                (gr->histogram[i])*=1/mean_last_frac;
        }
/*
        double newneigh=0;
        for (i=0;i<(gr->dimh);i++)
        {
                newneigh+=(gr->histogram[i])*((gr->bin)*(i+0.5))*((gr->bin)*(i+0.5))*4*M_PI*(gr->bin);
        }
        double coeff=1/(newneigh)*(box[0]*box[3]*box[5]);
        for (i=0;i<(gr->dimh);i++)
        {
                (gr->histogram[i])=(gr->histogram[i])*coeff;
        }
*/
        double newneigh=0;
        for (i=0;i<(gr->dimh);i++)
        {
                newneigh+=(gr->histogram[i])*((gr->bin)*(i+0.5))*((gr->bin)*(i+0.5))*4*M_PI*(gr->bin)*numparticles/(box[0]*box[3]*box[5]);
        }
        //printf("\nNew number of neighbors %lf\n",newneigh);


}

void grPrint(grstruct *gr,FILE *p_file)
{
	int i;
	double r;

	for (i=0;i<(gr->dimh);i++)
	{
		r=(gr->bin)*(i+0.5);

		fprintf(p_file,"%lf %lf\n",r,gr->histogram[i]);
	}

}

 void grSave(grstruct *gr,FILE *p_file)
 {
         int i;
         double r;

         for (i=0;i<(gr->dimh);i++)
         {
                 r=(gr->bin)*(i+0.5);

                 fprintf(p_file,"%lf %lf\n",r,gr->histogram[i]);
         }

 }
void grFree(grstruct *gr)
{
	free(gr->histogram);
	free(gr->volume_shell);
	free(gr);
}
