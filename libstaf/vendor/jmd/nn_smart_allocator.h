#ifndef NN_SMART_H
#define NN_SMART_H

double** Matrix2Ddouble(int dim1,int dim2);
int** Matrix2Dint(int dim1,int dim2);
double** IrregularMatrix2Ddouble(int dim1,int* dim2,int factor);
int** IrregularMatrix2Dint(int dim1,int* dim2,int factor);


#endif
