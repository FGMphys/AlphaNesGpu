#ifndef CELL_LIST_H
#define CELL_LIST_H

typedef struct _listcell {
	int *HoC;                            // Head of Chain for linked list
	int *LinkedList;                     // linked list
	int NumberCells_x;                     // number of cells in one direction
	int NumberCells_y;
	int NumberCells_z;
	float CellSize_x;                     // cell edge size
	float CellSize_y;
	float CellSize_z;
	int *MyCell;
} listcell;

void celllistConstructor(FILE *config_file,vector *pos,int max_number_colloids,int *ncolloids,float box[],float *cutoff,interactionmap *interactionList,int *movedparticle);
void celllistFree();

listcell* getList(const float Box[],float cutoff,int num_particles);
void freeList(listcell *l);

void resetList(listcell *l);

void copyList(listcell *dst,listcell *src,int ncolloids);

void updateList(listcell *l,const vector *pos,int num);

void fullUpdateList(listcell *l,vector *pos,int num,const float Box[],float cutoff);

void changeCell(listcell *l,const vector *oldpos,const vector *newpos,int num);

listcell *selfList();

void addToList(listcell *l,const vector *pos,int num);
void removeFromList(listcell *l,int num);
void changeIdentityInList(listcell *l,int oldnum,int newnum);

void calculateLinksWithinCutoff(listcell *l,vector *pos,float Box[],float cutoff,interactionmap *im,int *num_links);

vector getCellSize(listcell *l);

void calculateExtendedInteractionMapWithCutoffDistance(listcell *l,interactionmap *im,interactionmap *ime,vector *pos,float Box[],float cutoff);
void calculateInteractionMapWithCutoffDistanceOrdered(listcell *l,interactionmap *ime,vector *pos,const float* Box,float cutoff);


int getParticleInteractionMap(listcell *l,vector *pos,int label,interactionmap *im,float Box[],float cutoff);
int getParticleInteractionMap_ImPos(listcell *l,vector *pos,int label,interactionmap *im,int im_pos,float Box[],float cutoff);
int calculateSystemInteractionMapExtended(listcell *l,interactionmap *ime,vector *pos,float Box[],float cutoff);
int calculateSystemInteractionMap(listcell *l,interactionmap *im,vector *pos,float Box[],float cutoff);

void celllistUpdateList();

int celllistCountParticles(listcell *l,int cell_label);
int celllistGetCellNumber(listcell *l,vector *pos);


#endif
