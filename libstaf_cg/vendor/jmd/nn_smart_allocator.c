
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "vector.h"

#include "nn_smart_allocator.h"

double** Matrix2Ddouble(int dim1,int dim2){

    int j;
    double** array;
    array=(double**)calloc((dim1),sizeof(double*));
    array[0]=(double*)calloc((dim1)*(dim2),sizeof(double));
    for (j=0;j<(dim1);j++)
    array[j]=array[0]+j*(dim2);
    return array;
}

int** Matrix2Dint(int dim1,int dim2){

    int j;
    int** array;
    array=(int**)calloc((dim1),sizeof(int*));
    array[0]=(int*)calloc((dim1)*(dim2),sizeof(int));
    for (j=0;j<(dim1);j++)
    array[j]=array[0]+j*(dim2);
    return array;
}


double** IrregularMatrix2Ddouble(int dim1,int* dim2,int factor){
    int j;
    int totaldim=0;
    double** array;
    for (int k=0;k<dim1;k++){
    totaldim=totaldim+dim2[k];
    }
    totaldim*=factor;
    array=(double**)calloc(dim1,sizeof(double*));
    array[0]=(double*)calloc(totaldim,sizeof(double));
    for (j=1;j<dim1;j++)
    array[j]=array[0]+dim2[j-1]*factor;
    return array;
}

int** IrregularMatrix2Dint(int dim1,int* dim2,int factor){
    int j;
    int totaldim=0;
    int** array;
    for (int k=0;k<dim1;k++){
    totaldim=totaldim+dim2[k];
    }
    totaldim*=factor;
    array=(int**)calloc(dim1,sizeof(int*));
    array[0]=(int*)calloc(totaldim,sizeof(int));
    for (j=1;j<dim1;j++)
    array[j]=array[0]+dim2[j-1]*factor;
    return array;
}
