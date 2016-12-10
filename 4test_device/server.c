#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include <cuda.h>
#include <builtin_types.h>

// This will output the proper CUDA error strings
// in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        fprintf(stderr,
                "CUDA Driver API error = %04d from file <%s>, line %i.\n",
                err, file, line );
        exit(-1);
    }
}

// --- global variables ----------------------------------------------------
CUdevice   device;
CUcontext  context;
CUmodule   module;
CUfunction function;
size_t     totalGlobalMem;

char       *module_file = (char*) "kernel.ptx";
char       *kernel_name = (char*) "matSum";


// --- functions -----------------------------------------------------------
void initCUDA()
{
    int deviceCount = 0;
    CUresult err = cuInit(0);
    int major = 0, minor = 0;

    if (err == CUDA_SUCCESS)
        checkCudaErrors(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "Error: no devices supporting CUDA\n");
        exit(-1);
    }

    // get first CUDA device
    checkCudaErrors(cuDeviceGet(&device, 0));
    char name[100];
    cuDeviceGetName(name, 100, device);
    printf("> Using device 0: %s\n", name);

    // get compute capabilities and the devicename
    checkCudaErrors( cuDeviceComputeCapability(&major, &minor, device) );
    printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

    checkCudaErrors( cuDeviceTotalMem(&totalGlobalMem, device) );
    printf("  Total amount of global memory:   %llu bytes\n",
           (unsigned long long)totalGlobalMem);
    printf("  64-bit Memory Address:           %s\n",
           (totalGlobalMem > (unsigned long long)4*1024*1024*1024L)?
           "YES" : "NO");

    err = cuCtxCreate(&context, 0, device);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error initializing the CUDA context.\n");
        cuCtxDetach(context);
        exit(-1);
    }

    err = cuModuleLoad(&module, module_file);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error loading the module %s\n", module_file);
        cuCtxDetach(context);
        exit(-1);
    }

    err = cuModuleGetFunction(&function, module, kernel_name);

    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error getting kernel function %s\n", kernel_name);
        cuCtxDetach(context);
        exit(-1);
    }
}

void init_shared_memory()
{
	int shmid;
	char *shm, *s;
    key_t key = 5678;
	int SHMSZ = 2 + sizeof(struct CUctx_st);

    if ((shmid = shmget(key, SHMSZ, IPC_CREAT | 0666)) < 0) {
        perror("shmget");
        exit(1);
    }

    if ((shm = (char *)shmat(shmid, NULL, 0)) == (char *) -1) {
        perror("shmat");
        exit(1);
    }

    s = shm;
	memcpy(s, (void *)context, sizeof(struct CUctx_st));
	*(s+sizeof(struct CUctx_st)) = 'a';
}

void finalizeCUDA()
{
    cuCtxDetach(context);
}

void setupDeviceMemory(CUdeviceptr *d_a)
{
    checkCudaErrors( cuMemAlloc(d_a, sizeof(int)) );
}

void releaseDeviceMemory(CUdeviceptr d_a)
{
    checkCudaErrors( cuMemFree(d_a) );
}

void runKernel(CUdeviceptr d_a)
{
	int i = 2;
    void *args[2] = { &d_a, &i};

    // grid for kernel: <<<N, 1>>>
    checkCudaErrors( cuLaunchKernel(function, 1, 1, 1,  // Nx1x1 blocks
                                    1, 1, 1,            // 1x1x1 threads
                                    0, 0, args, 0) );
}

int main(int argc, char **argv)
{
    int a = 1;
    CUdeviceptr d_a;
	int i;

    // initialize
    printf("- Initializing...\n");
    initCUDA();

	init_shared_memory();

    // allocate memory
    setupDeviceMemory(&d_a);

    // copy arrays to device
    checkCudaErrors( cuMemcpyHtoD(d_a, &a, sizeof(int)) );

    // run
    printf("# Running the kernel...\n");
    runKernel(d_a);
    printf("# Kernel complete.\n");
	cudaDeviceSynchronize();

    // copy results to host and report
    //checkCudaErrors( cuMemcpyDtoH(c, d_c, sizeof(int) * N) );
    printf("*** All checks complete.\n");


    // finish
    printf("- Finalizing...\n");
    releaseDeviceMemory(d_a);
    finalizeCUDA();
    return 0;
}
