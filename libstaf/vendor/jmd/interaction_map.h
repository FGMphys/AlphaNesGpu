#ifndef INTERACTION_MAP_H
#define INTERACTION_MAP_H

#include "vector.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _interactionmap {
	int num;
	int *who;
	int **with;
	int *howmany;
	int num_bonds;
	int size;
	double **rij2;
	vector **rij;
} interactionmap;

interactionmap* createInteractionMap(int max_elements,int max_neighbours);
void freeInteractionMap(interactionmap *i);
void resetInteractionMap(interactionmap *i);
void buildImeFromIm(interactionmap *im,interactionmap *ime);

#ifdef __cplusplus
}
#endif

#endif
