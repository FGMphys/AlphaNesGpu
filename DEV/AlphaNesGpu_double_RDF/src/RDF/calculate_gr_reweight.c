#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "vector.h"
//#include "global_definitions.h"
//#include "log.h"
//#include "smart_allocator.h"
//#include "io.h"


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
void from_car_to_int(vector* ipos,const double* positions,int numparticle,double* box);

void grSample(grstruct *gr,vector *pos,int numparticles,double box[],const int* type_map,int nt,int nt2,const int* type,double delta_E);
void grAverage(grstruct *gr,int numparticles,double box[]);

void grFree(grstruct *gr);

void grPrint(grstruct *gr,FILE *p_file);
void grSave(grstruct *gr,FILE *p_file);
void compute_rdf(const double* energy,const double* position,const double* box_glob, int nf, int N,
    const double* ground_energy,const int* type_pair,const int* type, 
    const int* type_map,const double* binsize,const double* betarewe,double* RDF){



	double bin_size=binsize[0];

	int numtypes[1];

	int num_files=nf;


    const double* delta_E_arr=ground_energy;



	//////////////////////////////////////////////////////////
	double min_box_size=10000000.;
	// dobbiamo costruire la struttura per il calcolo del gr
	// potenzialmente le configurazioni vengono da simulazioni NpT
	// in cui il lato della box cambia. Fissiamo come range la meta'
	// del lato piu' piccolo
	double ibox[6],box[6];
	int numparticles=N;
	vector* ipos=(vector*)calloc(N,sizeof(vector));
    int old_numparticles;
	int i;


	old_numparticles=0;
	double average_density=0.;

//Computing average density of dataset
	for (i=0;i<num_files;i++)
	{
            memcpy(box,&box_glob[i*6],6*sizeof(double));

/*		if ((numparticles!=old_numparticles) && (first_file==0))
		{
			logPrint("Configurations have different number of particles\n");
			exit(1);
		}
*/
		average_density+=(double)numparticles/(box[0]*box[3]*box[5]);

		if (box[0]<min_box_size)
			min_box_size=box[0];
		if (box[3]<min_box_size)
			min_box_size=box[3];
		if (box[5]<min_box_size)
			min_box_size=box[5];


		old_numparticles=numparticles;
	}

	average_density/=num_files;

//Building gr for given pair

          	
		int nt=type_pair[0];
		int nt2=type_pair[1];
		grstruct *gr=grConstruct(bin_size,0.5*min_box_size);
		for (i=0;i<num_files;i++)
			{
//                      memcpy(&pos[0].x,&position[i*N*3],3*N*sizeof(double));
                        memcpy(box,&box_glob[i*6],6*sizeof(double));
                        from_car_to_int(ipos,&position[i*N*3],N,box);
	                double delta_E=-delta_E_arr[i]+energy[i]*betarewe[0];
		        //printf("%lf\n",delta_E);	
			grSample(gr,ipos,numparticles,box,type_map,nt,nt2,type,delta_E);
		
			}
        

	grAverage(gr,type[nt2],box);
	int steppy=0;
	char filename[100];
	sprintf(filename,"gr_%d_%d_rewe_step_%d.dat",nt,nt2,steppy);
	FILE *pfile=fopen(filename,"w");
	grSave(gr,pfile);
	
        for (i=0;i<(gr->dimh);i++)
         {
                 double r=(gr->bin)*(i+0.5);

                 RDF[i]=gr->histogram[i];
         }
	grFree(gr);
	fclose(pfile);
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

void from_car_to_int(vector* ipos,const double* positions,int numparticle,double* box){
     double Inobox[6];
     Inobox[0]=1./box[0];
     Inobox[1]=-box[1]/(box[0]*box[3]);
     Inobox[2]=(box[1]*box[4])/(box[0]*box[3]*box[5])-box[2]/(box[0]*box[5]);
     Inobox[3]=1./box[3];
     Inobox[4]=-box[4]/(box[3]*box[5]);
     Inobox[5]=1./box[5];
     for (int par=0;par<numparticle;par++){
	 double px=positions[3*par];
	 double py=positions[3*par+1];
	 double pz=positions[3*par+2];
         ipos[par].x=(Inobox[0]*px+Inobox[1]*py+Inobox[2]*pz);
         ipos[par].y=(Inobox[3]*py+Inobox[4]*pz);
         ipos[par].z=(Inobox[5]*pz); 
      }
};

void grSample(grstruct *gr,vector *pos,int numparticles,double Box[],const int* type_map,int nt,int nt2,const int* type,double delta_E)
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
        double newneigh=0;
        for (i=0;i<(gr->dimh);i++)
        {
                newneigh+=(gr->histogram[i])*((gr->bin)*(i+0.5))*((gr->bin)*(i+0.5))*4*M_PI*(gr->bin)*numparticles/(box[0]*box[3]*box[5]);
        }
        printf("\nNew number of neighbors %lf\n",newneigh);
*/
        
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
