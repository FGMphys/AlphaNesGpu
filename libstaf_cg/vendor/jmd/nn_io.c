#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "vector.h"
#include "global_definitions.h"
#include "io.h"
#include "nn_io.h"

#define NULLKEYWORD "none"


void fill_with_ones(double* vec,int length){
	   for (int k=0;k<length;k++){
			   vec[k]=1.;
		 }
}
void read_tipos(int* tipos,int numtypes,char* root_path)
{
	char line[MAX_LINE_LENGTH]="";
	char filename[4096];
	sprintf(filename,"%s",root_path);
  printf("Alphanes: searching for type.dat file in %s\n",filename);
  fflush(stdout);
	FILE *pfile=fopen(filename,"r");
        if (pfile==NULL){
	printf("\n No atom type file found in %s!\n",filename);
	exit(0);
       }
	int i=0;
	while (getLine(line,pfile)!=0)
	{
	sscanf(line,"%d",tipos+i);
	i++;
	}
	assert(numtypes==i);
  fclose(pfile);
}

void read_typemap(int* type_map,int num_of_particles,char* root_path)
{
	char line[MAX_LINE_LENGTH]="";
	char filename[4096];
	sprintf(filename,"%s",root_path);
  printf("Alphanes: searching for type_map.dat file in %s\n",filename);
  fflush(stdout);
	FILE *pfile=fopen(filename,"r");
	int i=0;
	while (getLine(line,pfile)!=0)
	{
	sscanf(line,"%d",type_map+i);
	i++;
	}
	assert(num_of_particles==i);
  fclose(pfile);
}



void readAlpha_num(char* root_path,int* alpha_num,int* alpha_a_num,int num_types){
  //char line[MAX_LINE_LENGTH]="";
  char filename[4096];
  for (int type=0;type<num_types;type++){
		  sprintf(filename,"%s/type%d_alpha_2body.dat",root_path,type);
		  FILE *pfile=fopen(filename,"r");

		  if (pfile == NULL) {
		      printf("Errore nell'apertura del file %s/type%d_alpha_2body.dat",root_path,type);
		      exit(EXIT_FAILURE);
		  }

		  // Determina le dimensioni della matrice
		  int colonne = 0;
                  int righe = 0; 
		  char riga[1000000];
		  while (fgets(riga, sizeof(riga), pfile) != NULL) {
                         righe++;
                        if (colonne == 0) {
                           char *token = strtok(riga, " ");
                           while (token != NULL) {
                                 colonne++;
                                 token = strtok(NULL, " ");
                                }
                         }
                      }

                assert(righe==3);
		alpha_num[type]=colonne;
	        fclose(pfile);
		}

  char filename_2[4096];
  //char line_2[MAX_LINE_LENGTH]="";
  int nt_couple=num_types*(num_types+1)/2;
  for (int type=0;type<num_types;type++){
		  sprintf(filename_2,"%s/type%d_alpha_3body.dat",root_path,type);
		  FILE *pfile_2=fopen(filename_2,"r");

			if (pfile_2 == NULL) {
		           printf("Errore nell'apertura del file %s/type%d_alpha_3body.dat",root_path,type);
		      exit(EXIT_FAILURE);
		  }
                  // Determina le dimensioni della matrice
                  int colonne = 0;
                  int righe = 0;
                  char riga[1000000];
                  while (fgets(riga, sizeof(riga), pfile_2) != NULL) {
                         righe++;
                        if (colonne == 0) {
                           char *token = strtok(riga, " ");
                           while (token != NULL) {
                                 colonne++;
                                 token = strtok(NULL, " ");
                                }
                         }
                      }

                  assert(righe==6);
		  alpha_a_num[type]=colonne/3;
		  fclose(pfile_2);
       }
}
void readalpha2b(int type,double *weights,int righe,int colonne,char* root_path){
			char filename[4096];
			sprintf(filename,"%s/type%d_alpha_2body.dat",root_path,type);
			FILE *pfile=fopen(filename,"r");


			for (int i = 0; i < righe; i++) {
		 			for (int j = 0; j < colonne; j++) {
				 			fscanf(pfile, "%lf", &weights[i*colonne+j]);
		      }
      }
      fclose(pfile);

}

void reademb2b(int type,double *weights,int righe,int colonne,char* root_path){
			//char line[MAX_LINE_LENGTH]="";
			char filename[4096];
			sprintf(filename,"%s/type%d_type_emb_2b_sq.dat",root_path,type);
			FILE *pfile=fopen(filename,"r");


			for (int i = 0; i < righe; i++) {
		 			for (int j = 0; j < colonne; j++) {
				 			fscanf(pfile, "%lf", &weights[i*colonne+j]);
		      }
      }
      fclose(pfile);

}

void reademb3b(int type,double *weights,int righe,int colonne,char* root_path){
			//char line[MAX_LINE_LENGTH]="";
			char filename[4096];
			sprintf(filename,"%s/type%d_type_emb_3b_sq.dat",root_path,type);
			FILE *pfile=fopen(filename,"r");

			for (int i = 0; i < righe; i++) {
		 			for (int j = 0; j < colonne; j++) {
				 	    fscanf(pfile, "%lf", &weights[i*colonne+j]);
		      }
      }
      fclose(pfile);

}

void readalpha3b(int type,double *weights,int righe,int colonne,char* root_path)
{
			//char line[MAX_LINE_LENGTH]="";
			char filename[4096];
			sprintf(filename,"%s/type%d_alpha_3body.dat",root_path,type);
			FILE *pfile=fopen(filename,"r");

			for (int i = 0; i < righe; i++) {
		 			for (int j = 0; j < colonne*3; j++) {
				 			fscanf(pfile, "%lf", &weights[i*3*colonne+j]);
		      }
      }
      fclose(pfile);
}

void print_data_int(int* data,int dim){
    printf("Alphanes_check: vector is");
  for (int k=0;k<dim;k++){
        printf(" %d ",data[k]);
        if (k>10)
        printf("\n");
}
printf("\n");
fflush(stdout);
}


void printok(){
  printf("\nQui Ok!\n");
  fflush(stdout);
}

void readType(char *initial_conditions_type,int *Type,double *Masses,int N)
{
  FILE *pfile=fopen(initial_conditions_type,"r");
  if (pfile==NULL)
     {
	     printf("\nNo masses vector found in %s!\n",initial_conditions_type);
             exit(0);

     }
  int n=0;

  while(fscanf(pfile,"%d %lf\n",Type+n,Masses+n)!=EOF)
  {
    n++;
  }


  fclose(pfile);
  assert(n==N);

}

void print_data_double(double* matrix,int righe ,int colonne){
     int j,k;
     printf("\nAlphanes_chek:\n");
     for (j=0;j<righe;j++){
         for (k=0;k<colonne;k++){
         printf("%lf ",matrix[j*colonne+k]);
      }
      printf("\n");
     }
}
