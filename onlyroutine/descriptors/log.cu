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

static FILE *LogFile=NULL;
static int Aperto=0;
static int IsStdout=0;

int logStart(char *filename,char* mode)
{
	if (Aperto==1)
		return 1;
	
	LogFile=fopen(filename,mode);
	Aperto=1;
	return 0;
}

int logStartStdout()
{
	if (Aperto==1)
		return 1;
	
	IsStdout=1;
	Aperto=1;
	LogFile=stdout;
	return 0;
}

int logStartStderr()
{
	if (Aperto==1)
		return 1;
	
	IsStdout=1;
	Aperto=1;
	LogFile=stderr;
	return 0;
}

int logPrint(char *formato,...)
{
	if (Aperto==0)
	{
		return 1;
	}
	
	va_list args;
	va_start(args,formato);
	vfprintf(LogFile,formato,args);
	va_end(args);
	return 0;
}

int logFlush()
{
	if (Aperto)
	{
		fflush(LogFile);
		return 0;
	}
	return 1;
}

int logClose()
{
	if ((Aperto==1) && (IsStdout==0))
	{
		fclose(LogFile);
		Aperto=0;
		return 0;
	}
	else if (IsStdout==1)
		return 0;
	else
		return 1;
}

