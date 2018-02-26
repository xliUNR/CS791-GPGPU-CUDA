# Point to includes of local directories
CC=/usr/local/cuda-9.0/bin/nvcc
INCLUDES=-I../include
O_FILES=hellocuda.o knn.o

all: $(O_FILES)
	$(CC) -o PA3 $(O_FILES)

hellocuda.o: ../src/hellocuda.cu
	$(CC) -c ../src/hellocuda.cu -o hellocuda.o $(INCLUDES)

knn.o: ../src/knn.cu
	$(CC) -c ../src/knn.cu -o knn.o $(INCLUDES)

clean:
	rm -f *.o
	rm -f *~