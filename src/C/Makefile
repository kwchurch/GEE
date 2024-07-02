files=GEE

files=GEE hello

all: ${files}

clean: 
	rm ${files}; rm *.o

util.o: util.c util.h
	cc -c util.c

GEE: GEE.c util.o
	cc -o GEE GEE.c util.o

hello: hello.c
	cc -o hello hello.c



