#ifndef INTERACTION_MAP_H
#define INTERACTION_MAP_H

typedef struct _interactionmap {
	int num;		// number particles interacting
	int *who;		// particles interacting
	int **with;		// interacting partners
	int *howmany;	// number of interacting partners
	int num_bonds;
	int size;		// total number of particles
	double **rij2;
	vector **rij;
} interactionmap;


typedef struct _3bodyinteractionmap {
	int num;  // num triangles
	int **t_index;
	vector **t_dr;
	double **t_dr2;
} threebodyim;


interactionmap* createInteractionMap(int max_elements,int max_neighbours);
void freeInteractionMap(interactionmap *i);
void resetInteractionMap(interactionmap *i);
void buildImFromIme(interactionmap *ime,interactionmap *im);
void buildImeFromIm(interactionmap *im,interactionmap *ime);

threebodyim* create3BodyInteractionMap(int max_triangles);
void free3BodyInteractionMap(threebodyim *i);
int build3BodyMapfromInteractionMap(threebodyim *tm,interactionmap *im,vector *pos,vector *box,double cutoff);

void generateImeSecondShell(interactionmap *ime_f,interactionmap *ime_s);
//void generateImeSecondShell_dist2(interactionmap *ime_f,interactionmap *ime_s,vector *pos,int ncolloids,int *buffer);
void generateImeSecondShell_dist2(interactionmap *ime_f,interactionmap *ime_s,vector *pos,int ncolloids,int *buffer,double Box[6]);
#endif
