# Point to includes of local directories
CC=/usr/local/cuda-9.0/bin/nvcc
INCLUDES=-I../include
O_FILES=cutthreadmain.o cudafunctions.o
MC=/usr/local/
all: $(O_FILES)
	$(CC) -o PA4 $(O_FILES)

cutthreadmain.o: ../src/cutthreadmain.cu
	$(CC) -c ../src/cutthreadmain.cu -o cutthreadmain.o $(INCLUDES)

cudafunctions.o: ../src/cudafunctions.cu
	$(CC) -c ../src/cudafunctions.cu -o cudafunctions.o $(INCLUDES)

clean:
	rm -f *.o
	rm -f *~