# Point to includes of local directories
CC=/usr/local/cuda-9.0/bin/nvcc
INCLUDES=-I../include
O_FILES=hellocuda.o matrixMult.o

all: $(O_FILES)
	$(CC) -o PA2 $(O_FILES)

hellocuda.o: ../src/hellocuda.cu
	$(CC) -c ../src/hellocuda.cu -o hellocuda.o $(INCLUDES)

matrixMult.o: ../src/matrixMult.cu
	$(CC) -c ../src/matrixMult.cu -o add.o $(INCLUDES)

clean:
	rm -f *.o
	rm -f *~