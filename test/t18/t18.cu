#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    cudaStream_t stream;
    cudaError_t err;

    int leastPriority, greatestPriority;
    err = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

    err = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority);

    // Test cudaStreamGetPriority
    int priority;
    err = cudaStreamGetPriority(stream, &priority);

    // Test cudaStreamGetFlags
    unsigned int flags;
    err = cudaStreamGetFlags(stream, &flags);

    err = cudaStreamDestroy(stream);
}
