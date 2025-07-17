#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
//#include "log.h"
#include "utilities.h"


#ifndef BUFSIZ
#define BUFSIZ 4096
#endif

void fcopy(FILE *f1,FILE *f2)
{
    char            buffer[BUFSIZ];
    size_t          n;

    while ((n = fread(buffer, sizeof(char), sizeof(buffer), f1)) > 0)
    {
        if (fwrite(buffer, sizeof(char), n, f2) != n)
	{
            //logPrint("Error: file copy failed\n");
	    exit(1);
	}
    }
}


int compareInt(const void *node1,const void *node2)
{
	if ( (*(int*)node1) > (*(int*)node2) ) return 1;
	if ( (*(int*)node1) < (*(int*)node2) ) return -1;
	else
		return 0;
}

void insertionSort(int *v,int *length,int num)
{
	int l=*length;
	v[l]=num;
	while ((l>0) && (v[l]<v[l-1]))
	{
		int buffer;
		buffer=v[l];
		v[l]=v[l-1];
		v[l-1]=buffer;
		l--;
	}
	(*length)++;
}

int insertionSortNoDuplicates(int *v,int *length,int num)
{
	int i=0;
	
	while ((i<*length) && (v[i]<num))
	{
		i++;
	}
	if (i==*length)
	{
		v[i]=num;
		(*length)++;
		return 0;
	}
	else if (v[i]>num)
	{
		int j;
		for (j=*length;j>i;j--)
		{
			v[j]=v[j-1];
		}
		v[j]=num;
		(*length)++;
		return 0;
	}
	else
		return 1;
}


int insertionSort_dist2(int *number,int num_el,double *dist2,double dist2_el,int *length)
{
	int l=*length;
	number[l]=num_el;
	dist2[l]=dist2_el;
	while ((l>0) && (dist2[l]<dist2[l-1]))
	{
		double buffer_dist2;
		buffer_dist2=dist2[l];
		dist2[l]=dist2[l-1];
		dist2[l-1]=buffer_dist2;
		
		int buffer_num;
		buffer_num=number[l];
		number[l]=number[l-1];
		number[l-1]=buffer_num;
		
		l--;
	}
	(*length)++;
	
	return l;
}



void pbcNearestImage(vector *image,const vector *origin)
{
	vector olddist;
	
	olddist.x=image->x-origin->x;
	olddist.y=image->y-origin->y;
	olddist.z=image->z-origin->z;
	
	image->x-=rint(olddist.x);
	image->y-=rint(olddist.y);
	image->z-=rint(olddist.z);
}

