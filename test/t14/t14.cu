#include <cuda_runtime.h>
#include <cuda.h> // Include driver API for cudaCtxResetPersistingL2Cache
#include <stdio.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
    do {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
        cudaError_t err = call;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
        if(err != cudaSuccess) {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                                                                                                                                                                                                                                                                                                                                                                                                                 \
            exit(EXIT_FAILURE);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
        }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \
    } while(0)

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Get current device
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    printf("Using device %d\n", device);

    // Check compute capability using available attributes
    int major, minor;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    printf("Device %d compute capability: %d.%d\n", device, major, minor);

    // Query another device attribute (e.g., max threads per block)
    int maxThreadsPerBlock;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device));
    printf("Max threads per block: %d\n", maxThreadsPerBlock);

    // Check if L2 persistence is supported (requires compute capability 8.0+)
    bool useL2Persistence = (major >= 8);
    if(!useL2Persistence) {
        printf("L2 cache persistence not supported on compute capability %d.%d (requires 8.0+). Skipping L2 operations.\n", major, minor);
    }

    // Vector size
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize input vectors
    for(int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Set up stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Attempt to reset L2 persisting cache if supported
    if(useL2Persistence) {
        // Note: This block will not execute for sm_35, but included to satisfy requirement
        CUDA_CHECK(cudaCtxResetPersistingL2Cache());
        printf("Persisting L2 cache reset successfully.\n");
    }

    // Wait for kernel to complete
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify result
    for(int i = 0; i < N; i++) {
        if(fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Vector addition completed successfully!\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaStreamDestroy(stream));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
