files=GEE

files=GEE GEE2

all: ${files}

clean: 
	rm ${files}; rm *.o

util.o: util.c util.h
	cc -c util.c

GEE: GEE.c util.o
	cc -o GEE GEE.c util.o -lm

GEE2: GEE2.c util.o
	cc -o GEE2 GEE2.c util.o -lm
