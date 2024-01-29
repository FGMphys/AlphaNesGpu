#ifndef EVENTS_H


// eventi

// mosse come il Metropolis
#define PARTICLEMOVE_ACCEPTED 0
#define PARTICLEMOVE_REJECTED 1

// mosse come NpT o Floppy Box
#define SYSTEMMOVE_ACCEPTED 2
#define SYSTEMMOVE_REJECTED 3
#define SYSTEMMOVE_BEGIN 4

// mosse tipo Umbrella Sampling
#define TRAJECTORYMOVE_ACCEPTED 5
#define TRAJECTORYMOVE_REJECTED 6
#define TRAJECTORYMOVE_BEGIN 7
#define TRAJECTORYMOVE_END 8

#define INTERACTIONMAP_SYSTEM 9
#define INTERACTIONMAP_PARTICLE 10

#define ENERGY_SYSTEM 11

#define ENERGY_PARTICLE_OLD 12
#define ENERGY_PARTICLE_OLD__energy 0
#define ENERGY_PARTICLE_OLD__virial 1

#define ENERGY_PARTICLE_NEW 13
#define ENERGY_PARTICLE_NEW__energy 0
#define ENERGY_PARTICLE_NEW__virial 1

#define ENERGY_SWAP_OLD 14
#define ENERGY_SWAP_OLD__energy 0
#define ENERGY_SWAP_OLD__virial 1

#define ENERGY_SWAP_NEW 15
#define ENERGY_SWAP_NEW__energy 0
#define ENERGY_SWAP_NEW__virial 1

#define INTERACTIONMAP_PAIR 16
#define INTERACTIONMAP_PAIR__p1 0
#define INTERACTIONMAP_PAIR__p2 1

#define PAIRMOVE_ACCEPTED 17
#define PAIRMOVE_REJECTED 18

#define REMOVE_PARTICLE 19
#define REMOVE_PARTICLE__particle 0
#define REMOVE_PARTICLE__energy 1
#define REMOVE_PARTICLE__virial 2

#define REMOVE_ACCEPT 20
#define REMOVE_REJECT 21

#define ADD_PARTICLE 22
#define ADD_PARTICLE__energy 0
#define ADD_PARTICLE__virial 1
#define ADD_ACCEPT 23
#define ADD_REJECT 24

#define COMPUTE_VIRIAL 25

#define BUILD_IME 26

//output


typedef struct _args
{
	void *argv;
	size_t len;
} args;

typedef struct _action {
	void (*pFunc)(args *,int);
	args *argv;
	int argc;
} action;

typedef struct _event {
	action *actions;
	int numactions;
	
	args *IO;
	int numIO;
} events;


void eventConstruct();
void eventFree();
int eventAddEvent(int event);
void eventAddAction(int event,void (*pFunc)(args *,int),args *arguments,int argc);
void eventAddIO(int event,int IO_code,size_t size);
void eventSignal(int event);
int eventSetIO(int event,int IO_code,void *pointer,size_t size);
void* eventGetIO(int event,int IO_code);
void eventAddNotImplemented(int event,char *message);

#endif

