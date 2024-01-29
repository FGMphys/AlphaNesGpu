#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "vector.h"
#include "smart_allocator.h"
#include "utilities.h"
#include "interaction_map.h"

#define SQR(x) ((x)*(x))

interactionmap* createInteractionMap(int max_elements,int max_neighbours)
{
	interactionmap *i=(interactionmap*)malloc(sizeof(interactionmap));;

	i->num=0;
	i->who=(int*)calloc(max_elements,sizeof(int));
	i->howmany=(int*)calloc(max_elements,sizeof(int));
	i->size=max_elements;
	i->num_bonds=0;

	// la creazione degli array bidimensionali con un massimo numero di vicini
	// e' delicata. Dobbiamo lasciare dello spazio in coda a questi array nel
	// caso ci sia uno sforamento nel numero di vicini dell'ultima particella.
	// lasciamo in fondo max_elements spazi vuoti
	Matrix2DSafe(i->with,max_elements,max_neighbours,max_elements,int);
	Matrix2DSafe(i->rij2,max_elements,max_neighbours,max_elements,float);
	Matrix2DSafe(i->rij,max_elements,max_neighbours,max_elements,vector);
	return i;
}

void resetInteractionMap(interactionmap *i)
{
	i->num=0;
	i->num_bonds=0;
	int j;
	for (j=0;j<i->size;j++)
	{
		i->howmany[j]=0;
	}
}

void freeInteractionMap(interactionmap *i)
{
	free(i->who);
	Free2D(i->with);
	Free2D(i->rij);
	Free2D(i->rij2);
	free(i->howmany);
	free(i);
}

void buildImFromIme(interactionmap *ime,interactionmap *im)
{
	// semplicemente mettiamo in im le coppie (i,j) in cui j>i
	im->num=ime->num;
	im->size=ime->size;
	im->num_bonds=ime->num_bonds;

	int i,index,j;
	for (i=0;i<im->num;i++)
	{
		im->who[i]=ime->who[i];

		int howmany_i=0;

		for (index=0;index<ime->howmany[i];index++)
		{
			j=ime->with[i][index];

			if (j>i)
			{
				im->rij2[i][howmany_i]=ime->rij2[i][index];
				im->rij[i][howmany_i]=ime->rij[i][index];
				im->with[i][howmany_i++]=j;
			}
		}

		im->howmany[i]=howmany_i;
	}
}

void buildImeFromIm(interactionmap *im,interactionmap *ime)
{
	ime->num=im->num;
	ime->size=im->size;
	ime->num_bonds=im->num_bonds;

	int i,index,j;
	for (i=0;i<im->num;i++)
	{
		ime->who[i]=im->who[i];

		for (index=0;index<im->howmany[i];index++)
		{
			j=im->with[i][index];
			float rij2=im->rij2[i][index];
			vector rij=im->rij[i][index];

			ime->rij[i][ime->howmany[i]]=rij;
			(ime->rij[j][ime->howmany[j]]).x=-rij.x;
			(ime->rij[j][ime->howmany[j]]).y=-rij.y;
			(ime->rij[j][ime->howmany[j]]).z=-rij.z;

			ime->rij2[i][ime->howmany[i]]=rij2;
			ime->rij2[j][ime->howmany[j]]=rij2;

			ime->with[i][ime->howmany[i]++]=j;
			ime->with[j][ime->howmany[j]++]=i;
		}
	}
}

threebodyim* create3BodyInteractionMap(int max_triangles)
{
	threebodyim *i=(threebodyim*)malloc(sizeof(threebodyim));;

	i->num=0;
	Matrix2D(i->t_index,max_triangles,3,int);
	Matrix2D(i->t_dr,max_triangles,3,vector);
	Matrix2D(i->t_dr2,max_triangles,3,float);

	return i;
}

void free3BodyInteractionMap(threebodyim *i)
{
	Free2D(i->t_index);
	Free2D(i->t_dr);
	Free2D(i->t_dr2);
	free(i);
}

int build3BodyMapfromInteractionMap(threebodyim *tm,interactionmap *im,vector *pos,vector *box,float cutoff)
{
	int i,j,k;

	float cutoff2=cutoff*cutoff;

	tm->num=0;

	for (i=0;i<im->num;i++)
	{
		int particle_i=im->who[i];

		for (j=0;j<im->howmany[i]-1;j++)
		{
			int particle_j=im->with[i][j];

			for (k=j+1;k<im->howmany[i];k++)
			{
				int particle_k=im->with[i][k];

				// attenzione non e' vero, i triangoli non vanno chiusi per forza
				// per chiudere il triangolo dobbiamo solo controllare se particle_k e particle_j sono vicine

				// probabilmente la cosa piu' veloce e' calcolare la distanza anziche' cercare tra i
				// vicini di j e k
				vector dist;
				dist.x=pos[particle_k].x-pos[particle_j].x;
				dist.y=pos[particle_k].y-pos[particle_j].y;
				dist.z=pos[particle_k].z-pos[particle_j].z;

				dist.x-=box->x*rint(dist.x/box->x);
				dist.y-=box->y*rint(dist.y/box->y);
				dist.z-=box->z*rint(dist.z/box->z);

				float dist2=SQR(dist.x)+SQR(dist.y)+SQR(dist.z);

				if (dist2<cutoff2)
				{
					// triangolo chiuso
					int num_triangle=tm->num;
					tm->t_index[num_triangle][0]=particle_i;
					tm->t_index[num_triangle][1]=particle_j;
					tm->t_index[num_triangle][2]=particle_k;

					tm->t_dr2[num_triangle][0]=im->rij2[i][j];
					tm->t_dr2[num_triangle][1]=im->rij2[i][k];
					tm->t_dr2[num_triangle][2]=dist2;

					tm->t_dr[num_triangle][0]=im->rij[i][j];
					tm->t_dr[num_triangle][1]=im->rij[i][k];
					tm->t_dr[num_triangle][2]=dist;

					(tm->num)++;
				}

			}


		}
	}

	return tm->num;
}

void generateImeSecondShell(interactionmap *ime_f,interactionmap *ime_s)
{
	int i;

	ime_s->num=ime_f->num;

	for (i=0;i<ime_f->num;i++)
	{
		ime_s->who[i]=ime_f->who[i];

		int numneighbour=0;

		int j;
		for (j=0;j<ime_f->howmany[i];j++)
		{
			// neighbours of i
			int n=ime_f->with[i][j];

			insertionSortNoDuplicates(ime_s->with[i],&numneighbour,n);

			// neighbours of n
			int k;
			for (k=0;k<ime_f->howmany[n];k++)
			{
				int nn=ime_f->with[n][k];

				if (nn!=i)
					insertionSortNoDuplicates(ime_s->with[i],&numneighbour,nn);
			}
		}

		ime_s->howmany[i]=numneighbour;

	}

}


// questa funzione e' sbagliata, non inserisce i vicini in maniera ordinata
/*
void generateImeSecondShell_dist2(interactionmap *ime_f,interactionmap *ime_s,vector *pos,int ncolloids,int *buffer)
{
	int i;

	ime_s->num=ime_f->num;


	for (i=0;i<ime_f->num;i++)
	{
		int nbuffer=0;

		ime_s->who[i]=ime_f->who[i];
		ime_s->howmany[i]=0;

		int j;
		for (j=0;j<ime_f->howmany[i];j++)
		{
			int n=ime_f->with[i][j];

			if (insertionSortNoDuplicates(buffer,&nbuffer,n)==0)
			{
				insertionSort_dist2(ime_s->with[i],n,ime_s->rij2[i],ime_f->rij2[i][j],ime_s->howmany+i);
			}


			// neighbours of n
			int k;
			for (k=0;k<ime_f->howmany[n];k++)
			{
				int nn=ime_f->with[n][k];

				if ((nn!=i) && (insertionSortNoDuplicates(buffer,&nbuffer,nn)==0))
				{
					insertionSort_dist2(ime_s->with[i],nn,ime_s->rij2[i],ime_f->rij2[n][k],ime_s->howmany+i);
				}

			}
		}

	}
}
*/



int insertionSort_ime_dist2(interactionmap *ime,int particle,int neighbour,vector dist,float dist2)
{
	int l=ime->howmany[particle];
	ime->with[particle][l]=neighbour;
	ime->rij2[particle][l]=dist2;
	ime->rij[particle][l]=dist;

	while ((l>0) && (ime->rij2[particle][l]<ime->rij2[particle][l-1]))
	{
		float buffer_dist2;
		buffer_dist2=ime->rij2[particle][l];
		ime->rij2[particle][l]=ime->rij2[particle][l-1];
		ime->rij2[particle][l-1]=buffer_dist2;

		int buffer_num;
		buffer_num=ime->with[particle][l];
		ime->with[particle][l]=ime->with[particle][l-1];
		ime->with[particle][l-1]=buffer_num;

		vector buffer_dist;
		buffer_dist=ime->rij[particle][l];
		ime->rij[particle][l]=ime->rij[particle][l-1];
		ime->rij[particle][l-1]=buffer_dist;

		l--;
	}

	ime->howmany[particle]++;

	return l;
}


void generateImeSecondShell_dist2(interactionmap *ime_f,interactionmap *ime_s,vector *pos,int ncolloids,int *buffer,float Box[6])
{
	int i;

	ime_s->num=ime_f->num;


	for (i=0;i<ime_f->num;i++)
	{
		int nbuffer=0;

		ime_s->who[i]=ime_f->who[i];
		ime_s->howmany[i]=0;

		int j;
		for (j=0;j<ime_f->howmany[i];j++)
		{
			int n=ime_f->with[i][j];

			if (insertionSortNoDuplicates(buffer,&nbuffer,n)==0)
			{
				insertionSort_ime_dist2(ime_s,i,n,ime_f->rij[i][j],ime_f->rij2[i][j]);
			}


			// neighbours of n
			int k;
			for (k=0;k<ime_f->howmany[n];k++)
			{
				int nn=ime_f->with[n][k];

				if ((nn!=i) && (insertionSortNoDuplicates(buffer,&nbuffer,nn)==0))
				{
					vector olddist,dist;

					olddist.x=pos[i].x-pos[nn].x;
					olddist.y=pos[i].y-pos[nn].y;
					olddist.z=pos[i].z-pos[nn].z;

					olddist.x-=rint(olddist.x);
					olddist.y-=rint(olddist.y);
					olddist.z-=rint(olddist.z);

					dist.x=Box[0]*olddist.x+Box[1]*olddist.y+Box[2]*olddist.z;
					dist.y=Box[3]*olddist.y+Box[4]*olddist.z;
					dist.z=Box[5]*olddist.z;

					float dist2=SQR(dist.x)+SQR(dist.y)+SQR(dist.z);

					insertionSort_ime_dist2(ime_s,i,nn,dist,dist2);

				}
			}
		}

	}
}

