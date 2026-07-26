#ifndef IO_H
#define IO_H

#define MAX_LINE_LENGTH 2000

int getLine(char *line,FILE *pfile);
int getNumberParticles(char *_input_name);
void getBoxLength(char *_input_name, double Box[]);
void getHeader(char *_input_name,steps *_step,int *_numparticles,double Box[]);
void readPositions(char *_input_name,vector *pos,steps *time,int *numparticles,double Box[],double INOBox[]);
void readVelocities(char *_input_name,vector *vel,steps *time,int *numparticles,double *box);
void savePositions(char *output_name,vector *pos,steps time,int numparticles,double Box[]);
void saveVelocities(char *output_name,vector *vel,steps time,int numparticles,double* box);
void saveForces(char *output_name,vector *forces,steps time,int numparticles,double* box);

#endif
