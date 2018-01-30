# Point to includes of local directories
CC=/usr/local/cuda-9.0/bin/nvcc
INCLUDES=-I../include
O_FILES=arraypop.o

all: $(O_FILES)
	$(CC) -o PA1 $(O_FILES)

arraypop.o: ../src/arraypop.cu
	$(CC) -c ../src/arraypop.cu -o arraypop.o $(INCLUDES)

clean:
	rm -f *.o
	rm -f *~