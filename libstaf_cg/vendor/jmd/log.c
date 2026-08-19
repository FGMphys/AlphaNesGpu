#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include "log.h"

/*
TODO:

*) introdurre una variabile per il controllo del flush bufferizzato

*) introdurre un puntatore alla variabile timestep in modo tale da poter avere delle funzioni
di log temporizzate. Qualcosa del tipo

logPrint( ...,500);  -> stampa al passo 500

*/

FILE *LogFile=NULL;
int Open=0;

void logStart(char *filename,char* mode)
{
	LogFile=fopen(filename,mode);
	Open=1;
}

void logPrint(char *formato,...)
{
	if (Open==0)
	{
		printf("Warning: no log file opened\n");
		return;
	}
	
	va_list args;
	va_start(args,formato);
	vfprintf(LogFile,formato,args);
	va_end(args);
}

void logFlush()
{
	if (Open)
		fflush(LogFile);
}

void logClose()
{
	fclose(LogFile);
	Open=0;
}

