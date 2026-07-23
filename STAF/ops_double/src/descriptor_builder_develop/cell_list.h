#ifndef CELL_LIST_H
#define CELL_LIST_H

typedef struct _listcell {
	int *HoC;                            // Head of Chain for linked list
	int *LinkedList;                     // linked list
	int NumberCells_x;                     // number of cells in one direction
	int NumberCells_y;
	int NumberCells_z;
	double CellSize_x;                     // cell edge size
	double CellSize_y;
	double CellSize_z;
	int *MyCell;
} listcell;

void celllistConstructor(FILE *config_file,vector *pos,int max_number_colloids,int *ncolloids,double box[],double *cutoff,interactionmap *interactionList,int *movedparticle);
void celllistFree();

listcell* getList(const double Box[],double cutoff,int num_particles);
void freeList(listcell *l);

void resetList(listcell *l);

void copyList(listcell *dst,listcell *src,int ncolloids);

void updateList(listcell *l,const vector *pos,int num);

void fullUpdateList(listcell *l,vector *pos,int num,const double Box[],double cutoff);

void changeCell(listcell *l,const vector *oldpos,const vector *newpos,int num);

listcell *selfList();

void addToList(listcell *l,const vector *pos,int num);
void removeFromList(listcell *l,int num);
void changeIdentityInList(listcell *l,int oldnum,int newnum);

void calculateLinksWithinCutoff(listcell *l,vector *pos,double Box[],double cutoff,interactionmap *im,int *num_links);

vector getCellSize(listcell *l);

void calculateExtendedInteractionMapWithCutoffDistance(listcell *l,interactionmap *im,interactionmap *ime,vector *pos,double Box[],double cutoff);
void calculateInteractionMapWithCutoffDistanceOrdered(listcell *l,interactionmap *ime,vector *pos,const double* Box,double cutoff);


int getParticleInteractionMap(listcell *l,vector *pos,int label,interactionmap *im,double Box[],double cutoff);
int getParticleInteractionMap_ImPos(listcell *l,vector *pos,int label,interactionmap *im,int im_pos,double Box[],double cutoff);
int calculateSystemInteractionMapExtended(listcell *l,interactionmap *ime,vector *pos,double Box[],double cutoff);
int calculateSystemInteractionMap(listcell *l,interactionmap *im,vector *pos,double Box[],double cutoff);

void celllistUpdateList();

int celllistCountParticles(listcell *l,int cell_label);
int celllistGetCellNumber(listcell *l,vector *pos);


#endif
