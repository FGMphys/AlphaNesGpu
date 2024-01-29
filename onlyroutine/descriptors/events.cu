#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>

#include "log.h"
#include "events.h"

#define MAX_EVENTS 50
#define MAX_ACTIONS 20
#define MAX_IO 5

static events *EventList;
static int NumEvents;

static args *ParametersEvents;
static int NumParametersEvents;
static char Message[1000];


static void eventNotImplemented(args *argv,int argc)
{
	char *message=(char*)argv[0].argv;

	logPrint("Event not implemented: %s\n",message);
	exit(1);
}

void eventConstruct()
{

	EventList=(events*)calloc(MAX_EVENTS,sizeof(events));
	NumEvents=0;

	int i;
	for (i=0;i<MAX_EVENTS;i++)
	{
		EventList[i].actions=NULL;
		EventList[i].numactions=0;

		EventList[i].IO=NULL;
		EventList[i].numIO=0;
	}

	NumParametersEvents=1;
	ParametersEvents=(args*)calloc(NumParametersEvents,sizeof(args));
	ParametersEvents[0].argv=(void*)Message;

}

void eventFree()
{
	int i;
	for (i=0;i<MAX_EVENTS;i++)
	{
		if (EventList[i].actions!=NULL)
		{
			free(EventList[i].actions);
		}
		if (EventList[i].IO!=NULL)
		{
			int j;
			for (j=0;j<MAX_IO;j++)
			{
				if (EventList[i].IO[j].argv!=NULL)
					free(EventList[i].IO[j].argv);
			}

			free(EventList[i].IO);
		}
	}

	free(ParametersEvents);

	free(EventList);
}


int eventAddEvent(int event)
{
	if (event>=MAX_EVENTS)
	{
		logPrint("Error: increase MAX_EVENTS\n");
		exit(1);
	}

	if ((EventList[event].actions!=NULL) || (EventList[event].IO!=NULL))
		return 0;
	else
	{
		EventList[event].actions=(action*)calloc(MAX_ACTIONS,sizeof(action));

		EventList[event].IO=(args*)calloc(MAX_IO,sizeof(args));

		int i;
		for (i=0;i<MAX_ACTIONS;i++)
		{
			EventList[event].actions[i].pFunc=NULL;
			EventList[event].actions[i].argv=NULL;
			EventList[event].actions[i].argc=0;
		}
		for (i=0;i<MAX_IO;i++)
		{
			EventList[event].IO[i].argv=NULL;
			EventList[event].IO[i].len=0;
		}

		NumEvents++;

		return 1;
	}
}

void eventAddAction(int event,void (*pFunc)(args *,int),args *arguments,int argc)
{
	int na=EventList[event].numactions;

	if (na==MAX_ACTIONS)
	{
		logPrint("Error: increase MAX_ACTIONS\n");
		exit(1);
	}

	EventList[event].actions[na].pFunc=pFunc;
	EventList[event].actions[na].argv=arguments;
	EventList[event].actions[na].argc=argc;

	EventList[event].numactions++;
}

void eventAddNotImplemented(int event,char *message)
{
	int na=EventList[event].numactions;

	if (na==MAX_ACTIONS)
	{
		logPrint("Error: increase MAX_ACTIONS\n");
		exit(1);
	}

	strcpy(Message,message);

	EventList[event].actions[na].pFunc=eventNotImplemented;
	EventList[event].actions[na].argv=ParametersEvents;
	EventList[event].actions[na].argc=1;

	EventList[event].numactions++;
}

void eventAddIO(int event,int IO_code,size_t size)
{
	if (IO_code>=MAX_IO)
	{
		logPrint("Error: increase MAX_IO\n");
		exit(1);
	}

	EventList[event].IO[IO_code].argv=malloc(size);
	EventList[event].IO[IO_code].len=size;
	EventList[event].numIO++;
}


void eventSignal(int event)
{
	int i;
	for (i=0;i<EventList[event].numactions;i++)
	{
		(*(EventList[event].actions[i].pFunc))(EventList[event].actions[i].argv,EventList[event].actions[i].argc);
	}
}

int eventSetIO(int event,int IO_code,void *pointer,size_t size)
{
	if (size!=EventList[event].IO[IO_code].len)
	{
		return 1;
	}
	else
	{
		memcpy(EventList[event].IO[IO_code].argv,pointer,size);
		return 0;
	}
	//EventList[event].IO[IO_code].len=size;
}

void* eventGetIO(int event,int IO_code)
{
	return EventList[event].IO[IO_code].argv;
}
