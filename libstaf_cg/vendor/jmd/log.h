#ifndef LOG_H
#define LOG_H

void logStart(char *filename,char* mode);

void logPrint(char *formato,...);

void logFlush();

void logClose();

#endif
