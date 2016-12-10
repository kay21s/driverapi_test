all:
	nvcc --ptx kernel.cu
	gcc server.c -o server -lrt -L/usr/local/cuda/lib64 -lcudart -lcuda
	gcc client.c -o client -lrt -L/usr/local/cuda/lib64 -lcudart -lcuda

