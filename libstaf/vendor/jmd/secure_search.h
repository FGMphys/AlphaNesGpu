// SER  search module  //

/* Attention the search table is to be used within the
same function otherwise the keys are freed from memory
*/

/*
TO BE IMPLEMENTED: check for searched values in the correct range
*/


#ifndef SEARCH_H
#define SEARCH_H

#define MAXKEYS 100

typedef enum _Types {INT,ULLINT,ULINT,DOUBLE,STRING,MULTIPLE,CHAR} Types;


typedef union _typeunion {
	double *fvalue;
	int *ivalue;
	char *cvalue;
	unsigned long long int *ullivalue;
	unsigned long int *ulivalue;
} TypeUnion;


typedef struct _Value {
	Types type;
	TypeUnion pvalue;  // pointer to value
} Value;


typedef struct _SearchTable {
	int NumKeys;
	char *Keys[MAXKEYS];
	Value Values[MAXKEYS];
	int Found[MAXKEYS];
	int Compulsory[MAXKEYS];   // 1 compulsory key
} SearchTable;


SearchTable* searchNew();

/* Wish there were templates in C so we could have just one search
template function and keep type safety.
Something like: searchSearch<int> (...)
*/

void searchFree(SearchTable *);

void searchFile(FILE *,SearchTable *);

void searchDouble(char *,double *,SearchTable *);
int* searchTryDouble(char *,double *,SearchTable *);

void searchInt(char *,int *,SearchTable *);
int* searchTryInt(char *,int *,SearchTable *);

void searchChar(char *,char *,SearchTable *);
int* searchTryChar(char *,char *,SearchTable *);

void searchString(char *,char *,SearchTable *);
int* searchTryString(char *,char *,SearchTable *);

void searchMultiple(char *,char *,SearchTable *);
int* searchTryMultiple(char *,char *,SearchTable *);

void searchULLInt(char *,unsigned long long int *,SearchTable *);
int* searchTryULLInt(char *,unsigned long long int *,SearchTable *);

void searchULInt(char *,unsigned long int *,SearchTable *);
int* searchTryULInt(char *,unsigned long int *,SearchTable *);


// it returns the corresponding value of Compulsory to be checked after
// the search. If you change your mind and want the variable to be
// compulsory you can change to 1 this variable prior to search


#endif
