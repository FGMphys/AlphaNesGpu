#ifndef UTILITIES_H
#define UTILITIES_H

void fcopy(FILE *f1, FILE *f2);
int compareInt(const void *node1,const void *node2);
void insertionSort(int *v,int *length,int num);
void pbcNearestImage(vector *image,const vector *origin);
int insertionSortNoDuplicates(int *v,int *length,int num);
int insertionSort_dist2(int *number,int num_el,double *dist2,double dist2_el,int *length);

#endif
