#ifndef LOG_H
#define LOG_H

int logStart(char *filename,char* mode);

int logStartStdout();

int logStartStderr();

int logPrint(char *formato,...);

int logFlush();

int logClose();

#endif
