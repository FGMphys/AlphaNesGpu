#ifndef NN_IO_H
#define NN_IO_H

#define MAX_LINE_LENGTH 2000

//int getLine(char *line,FILE *pfile);
void readAlpha_num(char* root_path,int* alpha_num,int* alpha_a_num,int num_types);
void readalpha3b(int type,double *weights,int righe,int colonne,char* root_path);
void readalpha2b(int type,double *weights,int righe,int colonne,char* root_path);
void reademb3b(int type,double *weights,int righe,int colonne,char* root_path);
void reademb2b(int type,double *weights,int righe,int colonne,char* root_path);
void read_tipos(int* tipos,int numtypes,char* root_path);
void read_typemap(int* type_map,int num_of_particles,char* root_path);
//FILE* openFile(char filename[],char mod[]);
void print_data_int(int* data,int dim);
void print_data_doub(double* data,int dim);
void fill_with_ones(double* vec,int length);
void printok();
void readType(char *initial_conditions_type,int *Type,double *Masses,int N);
#endif
