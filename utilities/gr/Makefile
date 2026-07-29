# Fast multi-species g(r) (C cell-list + Python)
CC ?= gcc
CFLAGS ?= -O3 -fPIC -std=c11 -Wall -Wextra
CPPFLAGS += -Iinclude
LDFLAGS ?= -shared
LDLIBS ?= -lm

LIB = libstaf_gr.so
OBJS = src/staf_gr.o

.PHONY: all clean test

all: $(LIB)

$(LIB): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

src/staf_gr.o: src/staf_gr.c include/staf_gr.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ src/staf_gr.c

clean:
	rm -f $(OBJS) $(LIB)
