#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <assert.h>
#include "vector.h"
#include "global_definitions.h"
#include "log.h"
#include "io.h"


int getLine(char *line,FILE *pfile)
{
	if (fgets(line,MAX_LINE_LENGTH,pfile) == NULL)
		return 0;
	else
		return strlen(line);
}

int getNumberParticles(char *_input_name)
{
	FILE *ifile=fopen(_input_name,"r");

	if (ifile==NULL)
	{
		logPrint("Error: can't open initial conditions file '%s'\n",_input_name);
		exit(1);
	}

	int _numparticles;
	int state=0;
	char line[MAX_LINE_LENGTH]="";

	// HEADER: first line

	getLine(line,ifile);

	char stringa[500];
	strcpy(stringa,line);
	char *pch;
	int tokens=0;
	pch = strtok (stringa," ");
	while (pch != NULL)
	{
		tokens++;
		pch = strtok (NULL, " ");
	}

	if (tokens==8)
		state=sscanf(line,"%*d %d %*f %*f %*f %*f %*f %*f\n",&_numparticles);
	else if (tokens==5)
	{
		state=sscanf(line,"%*d %d %*f %*f %*f\n",&_numparticles);
	}



	if (state!=1)
	{
		logPrint("Error while reading number of particles in '%s' file\n",_input_name);
		exit(1);
	}

	fclose(ifile);

	return _numparticles;

}


void getBoxLength(char *_input_name, double Box[])
{
	FILE *ifile=fopen(_input_name,"r");
	
	if (ifile==NULL)
	{
		logPrint("Error: can't open initial conditions file '%s'\n",_input_name);
		exit(1);
	}
	
	int state=0;
	char line[MAX_LINE_LENGTH]="";
	
	
	
	
	getLine(line,ifile);
	
	char stringa[500];
	strcpy(stringa,line);
	char *pch;
	int tokens=0;
	pch = strtok (stringa," ");
	while (pch != NULL)
	{
		tokens++;
		pch = strtok (NULL, " ");
	}
	
	if (tokens==8)
		state=sscanf(line,"%*d %*d %lf %lf %lf %lf %lf %lf\n",Box+0,Box+1,Box+2,Box+3,Box+4,Box+5);
	else if (tokens==5)
	{
		state=sscanf(line,"%*d %*d %lf %lf %lf\n",Box+0,Box+3,Box+5);
		Box[1]=0.;
		Box[2]=0.;
		Box[4]=0.;
	}
	else
	{
		logPrint("Error while reading box length in '%s' file\n",_input_name);
		exit(1);
	}
	
	// check
	int ninversions=0;
	if (Box[0]<0)
	{
		Box[0]*=-1;
		ninversions++;
	}
	if (Box[3]<0)
	{
		Box[1]*=-1;
		Box[3]*=-1;
		ninversions++;
	}
	if (Box[5]<0)
	{
		Box[2]*=-1;
		Box[4]*=-1;
		Box[5]*=-1;
		ninversions++;
	}
	
	assert(ninversions%2==0);
	
	fclose(ifile);
}




void getHeader(char *_input_name,steps *_step,int *_numparticles,double Box[])
{
	FILE *ifile=fopen(_input_name,"r");
	
	if (ifile==NULL)
	{
		logPrint("Error: can't open initial conditions file '%s'\n",_input_name);
		exit(1);
	}
	
	int state=0;
	char line[MAX_LINE_LENGTH]="";
	
	// HEADER: first line
	
	getLine(line,ifile);
	
	char stringa[500];
	strcpy(stringa,line);
	char *pch;
	int tokens=0;
	pch = strtok (stringa," ");
	while (pch != NULL)
	{
		tokens++;
		pch = strtok (NULL, " ");
	}
	
	if (tokens==8)
		state=sscanf(line,"%lld %d %lf %lf %lf %lf %lf %lf\n",_step,_numparticles,Box+0,Box+1,Box+2,Box+3,Box+4,Box+5);
	else if (tokens==5)
	{
		state=sscanf(line,"%lld %d %lf %lf %lf\n",_step,_numparticles,Box+0,Box+3,Box+5);
		Box[1]=0.;
		Box[2]=0.;
		Box[4]=0.;
	}
	else
	{
		logPrint("Error while reading header of file %s\n",_input_name);
		exit(1);
	}
	
	// check
	int ninversions=0;
	if (Box[0]<0)
	{
		Box[0]*=-1;
		ninversions++;
	}
	if (Box[3]<0)
	{
		Box[1]*=-1;
		Box[3]*=-1;
		ninversions++;
	}
	if (Box[5]<0)
	{
		Box[2]*=-1;
		Box[4]*=-1;
		Box[5]*=-1;
		ninversions++;
	}
	
	assert(ninversions%2==0);
	
	fclose(ifile);
}

void readPositions(char *_input_name,vector *pos,steps *time,int *numparticles,double NOBox[],double INOBox[])
{
	FILE *ifile=fopen(_input_name,"r");
	
	if (ifile==NULL)
	{
		logPrint("Error: can't open initial conditions file '%s'\n",_input_name);
		exit(1);
	}
	
	int state=0;
	
	char line[MAX_LINE_LENGTH]="";
	
	// HEADER: first line
	
	// controlliamo il numero di lati di box specificati - se 3 impostiamo la scatola cubica
	getLine(line,ifile);
	
	char stringa[500];
	strcpy(stringa,line);
	char *pch;
	int tokens=0;
	pch = strtok (stringa," ");
	while (pch != NULL)
	{
		tokens++;
		pch = strtok (NULL, " ");
	}
	
	if (tokens==8)
		state=sscanf(line,"%lld %d %lf %lf %lf %lf %lf %lf\n",time,numparticles,NOBox+0,NOBox+1,NOBox+2,NOBox+3,NOBox+4,NOBox+5);
	else if (tokens==5)
	{
		state=sscanf(line,"%lld %d %lf %lf %lf\n",time,numparticles,NOBox+0,NOBox+3,NOBox+5);
		NOBox[1]=0.;
		NOBox[2]=0.;
		NOBox[4]=0.;
	}
	else
	{
		logPrint("Error while reading header of file %s\n",_input_name);
		exit(1);
	}
	
	// check
	int xinversion=1;
	int yinversion=1;
	int zinversion=1;

	int ninversions=0;
	if (NOBox[0]<0)
	{
		xinversion=-1;
		NOBox[0]*=-1;
		ninversions++;
	}
	if (NOBox[3]<0)
	{
		yinversion=-1;
		NOBox[1]*=-1;
		NOBox[3]*=-1;
		ninversions++;
	}
	if (NOBox[5]<0)
	{
		zinversion=-1;
		NOBox[2]*=-1;
		NOBox[4]*=-1;
		NOBox[5]*=-1;
		ninversions++;
	}
	
	assert(ninversions%2==0);
	
	
	INOBox[0]=1./NOBox[0];
	INOBox[1]=-NOBox[1]/(NOBox[0]*NOBox[3]);
	INOBox[2]=(NOBox[1]*NOBox[4])/(NOBox[0]*NOBox[3]*NOBox[5])-NOBox[2]/(NOBox[0]*NOBox[5]);
	INOBox[3]=1./NOBox[3];
	INOBox[4]=-NOBox[4]/(NOBox[3]*NOBox[5]);
	INOBox[5]=1./NOBox[5];
	
	
	//printf("%lf %lf %lf %lf %lf %lf\n",NOBox[0]*INOBox[0],NOBox[0]*INOBox[1]+NOBox[1]*INOBox[3],NOBox[0]*INOBox[2]+NOBox[1]*INOBox[4]+NOBox[2]*INOBox[5],NOBox[3]*INOBox[3],NOBox[3]*INOBox[4]+NOBox[4]*INOBox[5],NOBox[5]*INOBox[5]);
	//exit(1);
	
	int i;
	
	for (i=0;i<*numparticles;i++)
	{
		vector p;
		getLine(line,ifile);
		state=sscanf(line,"%lf %lf %lf\n",&(p.x),&(p.y),&(p.z));

		// leggiamo e convertiamo nella base non ortogonale
// 		pos[i].x=(double)xinversion*(INOBox[0]*p.x+INOBox[1]*p.y+INOBox[2]*p.z);
// 		pos[i].y=(double)yinversion*(INOBox[3]*p.y+INOBox[4]*p.z);
// 		pos[i].z=(double)zinversion*(INOBox[5]*p.z);
		pos[i].x=(INOBox[0]*p.x+INOBox[1]*p.y+INOBox[2]*p.z);
		pos[i].y=(INOBox[3]*p.y+INOBox[4]*p.z);
		pos[i].z=(INOBox[5]*p.z);
		
		if (state!=3)
		{
			logPrint("Error while reading '%s' file\n",_input_name);
			exit(1);
		}
	}
	
	fclose(ifile);
}


void readVelocities(char *_input_name,vector *vel,steps *time,int *numparticles,double *box)
{

	FILE *ifile=fopen(_input_name,"r");

	if (ifile==NULL)
	{
		logPrint("Error: can't open initial conditions file '%s'\n",_input_name);
		exit(1);
	}

	int state=0;

	char line[MAX_LINE_LENGTH]="";

	// HEADER: first line

	getLine(line,ifile);

	state=5;//sscanf(line,"%lld %d %lf %lf %lf\n",time,numparticles,&box->x,&box->y,&box->z);

	if (state!=5)
	{
		logPrint("Error while reading header in '%s' file\n",_input_name);
		exit(1);
	}

	int i;

	for (i=0;i<*numparticles;i++)
	{
		getLine(line,ifile);
		state=sscanf(line,"%lf %lf %lf\n",&(vel[i].x),&(vel[i].y),&(vel[i].z));

		if (state!=3)
		{
			logPrint("Error while reading '%s' file\n",_input_name);
			exit(1);
		}
	}

	fclose(ifile);
}


void savePositions(char *output_name,vector *pos,steps time,int numparticles,double Box[])
{
	FILE *ofile=fopen(output_name,"w");
	
	if (ofile==NULL)
	{
		logPrint("Error: can't open initial conditions file '%s'\n",output_name);
		exit(1);
	}
	
	// HEADER: first line
	fprintf(ofile,"%lld %d %lf %lf %lf %lf %lf %lf\n",time,numparticles,Box[0],Box[1],Box[2],Box[3],Box[4],Box[5]);
	
	int i;
	
	
	for (i=0;i<numparticles;i++)
	{
		vector p;
		
		
		// salviamo sempre nella base ortogonale
		p.x=Box[0]*pos[i].x+Box[1]*pos[i].y+Box[2]*pos[i].z;
		p.y=Box[3]*pos[i].y+Box[4]*pos[i].z;
		p.z=Box[5]*pos[i].z;
		
		
		fprintf(ofile,"%lf %lf %lf\n",p.x,p.y,p.z);
	}
	
	fclose(ofile);
}



void saveVelocities(char *output_name,vector *vel,steps time,int numparticles,double* box)
{
	FILE *ofile=fopen(output_name,"w");

	if (ofile==NULL)
	{
		logPrint("Error: can't open initial conditions file '%s'\n",output_name);
		exit(1);
	}

	// HEADER: first line
	fprintf(ofile,"%lld %d %lf %lf %lf %lf %lf %lf\n",time,numparticles,box[0],box[1],box[2],box[3],box[4],box[5]);

	int i;

	for (i=0;i<numparticles;i++)
	{

		fprintf(ofile,"%lf %lf %lf\n",vel[i].x,vel[i].y,vel[i].z);
	}

	fclose(ofile);
}


void saveForces(char *output_name,vector *forces,steps time,int numparticles,double* box)
{
	FILE *ofile=fopen(output_name,"w");

	if (ofile==NULL)
	{
		logPrint("Error: can't open initial conditions file '%s'\n",output_name);
		exit(1);
	}

	// HEADER: first line
        fprintf(ofile,"%lld %d %lf %lf %lf %lf %lf %lf\n",time,numparticles,box[0],box[1],box[2],box[3],box[4],box[5]);
	int i;

	for (i=0;i<numparticles;i++)
	{

		fprintf(ofile,"%lf %lf %lf\n",forces[i].x,forces[i].y,forces[i].z);
	}

	fclose(ofile);
}
