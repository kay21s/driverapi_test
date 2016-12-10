#include <stdio.h>

extern "C" __global__ void matSum(int *a, int i)
{
	int tid = blockIdx.x;
	if (threadIdx.x == 0) printf("my block id is %d, a is %d\n", tid, *a);

	clock_t start = clock();
	clock_t now;

	if (i == 2) {
		printf("i is 0\n");
		return;
	}

	for (;;) {
		now = clock();
		clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
		if (cycles >= 100000) {
			printf("A is %d\n", *a);
			start = clock();
			//break;
		}
	}
}
