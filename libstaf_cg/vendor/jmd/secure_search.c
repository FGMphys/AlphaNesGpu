#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>

#include "secure_search.h"

#define MAXLENGTH 100
#define MAX_LINE_LENGTH 200

/* Wish there were templates in C so we could have just one search
template function and keep type safety.
Something like: searchSearch<int> (...)
*/


//                          SEARCH FUNCTIONS                     //


static void searchFindSearchErrors(SearchTable *search)
{
	int i;
	int num=search->NumKeys;
	int troubles=0;
	int found;
	int compulsory;
	
	for (i=0;i<num;i++)
	{
		found=search->Found[i];
		compulsory=search->Compulsory[i];
		
		if ((found==0) && (compulsory))
		{
			troubles++;
			printf("Error: key '%s' not found\n",search->Keys[i]);
		}
		else if (found>1)
		{
			troubles++;
			printf("Error: key '%s' defined %d times\n",search->Keys[i],found);
		}
		
	}
	
	if (troubles>0)
		exit(1);
	
	
}

static int Getline(char *line,FILE *pfile)
{
	if (fgets(line,MAX_LINE_LENGTH,pfile) == NULL)
		return 0;
	else
		return strlen(line);
	
}


// performance can be improved by stop searching for keys as soon as they are found
void searchFile(FILE *pfile,SearchTable *search)
{
	char key[MAXLENGTH];               // key in file
	char value[MAXLENGTH];             // value in file
	
	int i;
	int num=search->NumKeys;
	char line[MAX_LINE_LENGTH];        // line of pfile
	
	rewind(pfile);
	
	
	
	while (!feof(pfile))
	{
		
// 		skip empty lines
		if (Getline(line,pfile))
		{
			
			strcpy(key,"");
			strcpy(value,"");
			
			if(sscanf(line,"%s = %s",key,value)!=2)
				continue;
			
			// scan through SearchTable keys
			
			i=0;
			
			while (i<num)
			{
				if (strcmp(search->Keys[i],key)==0)
				{
					
					search->Found[i]+=1;
					search->Compulsory[i]=1;
					
					switch (search->Values[i].type) {
						
						case INT:
							*(search->Values[i].pvalue.ivalue)=atoi(value);
							break;
							
						case ULLINT:
							*(search->Values[i].pvalue.ullivalue)=
								strtoull(value,(char**)NULL,10);
							
							break;
							
						case ULINT:
							*(search->Values[i].pvalue.ulivalue)=
									strtoul(value,(char**)NULL,10);
							
							break;
							
						case DOUBLE:
							*(search->Values[i].pvalue.fvalue)=atof(value);
							break;
							
							
						case STRING:
							strcpy(search->Values[i].pvalue.cvalue,value);
							break;
							
						case CHAR:
							memcpy(search->Values[i].pvalue.cvalue,value,sizeof(char));
							break;
						
						case MULTIPLE:
							;
							char *values=strstr(line,"=");
							
							values+=1;
							
							strcpy(search->Values[i].pvalue.cvalue,values);
							
							break;
							
						default:
							printf("Error: you should not be here! \n");
							break;
							
					}
					
					break;
				}
				else i++;
			}
		}
		
	}
	
	searchFindSearchErrors(search);
	
// 	POSSIBILITY
// 	reset search->Found for future searches with same table
	
}
SearchTable* searchNew()
{
	SearchTable *s=malloc(sizeof(SearchTable));
	s->NumKeys=0;
	
	return s;
}
void searchFree(SearchTable *s)
{
	free(s);
}


void searchInt(char *key,int *variable,SearchTable *search)
{
	int num=search->NumKeys;
	
	search->Values[num].type=INT;
	
	search->Values[num].pvalue.ivalue=variable;
	
	search->Keys[num]=key;
	
	search->Found[num]=0;
	
	search->Compulsory[num]=1;
	
	(search->NumKeys)++;
	
}

int* searchTryInt(char *key,int *variable,SearchTable *search)
{
	int num=search->NumKeys;
	
	search->Values[num].type=INT;
	
	search->Values[num].pvalue.ivalue=variable;
	
	search->Keys[num]=key;
	
	search->Found[num]=0;
	
	search->Compulsory[num]=0;
	
	(search->NumKeys)++;
	
	return search->Compulsory+num;
	
}


void searchULLInt(char *key,unsigned long long int *variable,SearchTable *search)
{
	int num=search->NumKeys;
	
	search->Values[num].type=ULLINT;
	
	search->Values[num].pvalue.ullivalue=variable;
	
	search->Keys[num]=key;
	
	search->Found[num]=0;
	
	search->Compulsory[num]=1;
	
	(search->NumKeys)++;
	
}


int* searchTryULLInt(char *key,unsigned long long int *variable,SearchTable *search)
{
	int num=search->NumKeys;
	
	search->Values[num].type=ULLINT;
	
	search->Values[num].pvalue.ullivalue=variable;
	
	search->Keys[num]=key;
	
	search->Found[num]=0;
	
	search->Compulsory[num]=0;
	
	(search->NumKeys)++;
	
	return search->Compulsory+num;
	
}

void searchULInt(char *key,unsigned long int *variable,SearchTable *search)
{
	int num=search->NumKeys;
	
	search->Values[num].type=ULINT;
	
	search->Values[num].pvalue.ulivalue=variable;
	
	search->Keys[num]=key;
	
	search->Found[num]=0;
	
	search->Compulsory[num]=1;
	
	(search->NumKeys)++;
	
}


int* searchTryULInt(char *key,unsigned long int *variable,SearchTable *search)
{
	int num=search->NumKeys;
	
	search->Values[num].type=ULINT;
	
	search->Values[num].pvalue.ulivalue=variable;
	
	search->Keys[num]=key;
	
	search->Found[num]=0;
	
	search->Compulsory[num]=0;
	
	(search->NumKeys)++;
	
	return search->Compulsory+num;
	
}


void searchDouble(char *key,double *variable,SearchTable *search)
{
	int num=search->NumKeys;
	
	search->Values[num].type=DOUBLE;
	
	search->Values[num].pvalue.fvalue=variable;
	
	search->Keys[num]=key;
	
	search->Found[num]=0;
	
	search->Compulsory[num]=1;
	
	(search->NumKeys)++;
	
}

int* searchTryDouble(char *key,double *variable,SearchTable *search)
{
	int num=search->NumKeys;
	
	search->Values[num].type=DOUBLE;
	
	search->Values[num].pvalue.fvalue=variable;
	
	search->Keys[num]=key;
	
	search->Found[num]=0;
	
	search->Compulsory[num]=0;
	
	(search->NumKeys)++;
	
	return search->Compulsory+num;
	
}

void searchString(char *key,char *variable,SearchTable *search)
{
	int num=search->NumKeys;
	
	search->Values[num].type=STRING;
	
	search->Values[num].pvalue.cvalue=variable;
	
	search->Keys[num]=key;
	
	search->Found[num]=0;
	
	search->Compulsory[num]=1;
	
	(search->NumKeys)++;
	
}


int* searchTryString(char *key,char *variable,SearchTable *search)
{
	int num=search->NumKeys;
	
	search->Values[num].type=STRING;
	
	search->Values[num].pvalue.cvalue=variable;
	
	search->Keys[num]=key;
	
	search->Found[num]=0;
	
	search->Compulsory[num]=0;
	
	(search->NumKeys)++;
	
	return search->Compulsory+num;
	
}

void searchChar(char *key,char *variable,SearchTable *search)
{
	int num=search->NumKeys;
	
	search->Values[num].type=CHAR;
	
	search->Values[num].pvalue.cvalue=variable;
	
	search->Keys[num]=key;
	
	search->Found[num]=0;
	
	search->Compulsory[num]=1;
	
	(search->NumKeys)++;
	
}


int* searchTryChar(char *key,char *variable,SearchTable *search)
{
	int num=search->NumKeys;
	
	search->Values[num].type=CHAR;
	
	search->Values[num].pvalue.cvalue=variable;
	
	search->Keys[num]=key;
	
	search->Found[num]=0;
	
	search->Compulsory[num]=0;
	
	(search->NumKeys)++;
	
	return search->Compulsory+num;
	
}



void searchMultiple(char *key,char *variable,SearchTable *search)
{
	int num=search->NumKeys;
	
	search->Values[num].type=MULTIPLE;
	
	search->Values[num].pvalue.cvalue=variable;
	
	search->Keys[num]=key;
	
	search->Found[num]=0;
	
	search->Compulsory[num]=1;
	
	(search->NumKeys)++;
	
}


int* searchTryMultiple(char *key,char *variable,SearchTable *search)
{
	int num=search->NumKeys;
	
	search->Values[num].type=MULTIPLE;
	
	search->Values[num].pvalue.cvalue=variable;
	
	search->Keys[num]=key;
	
	search->Found[num]=0;
	
	search->Compulsory[num]=0;
	
	(search->NumKeys)++;
	
	return search->Compulsory+num;
	
}

