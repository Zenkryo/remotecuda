#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Error checking macro
#define CUDA_CHECK(call)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
    do {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
        cudaError_t err = call;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
        if(err != cudaSuccess) {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                                                                                                                                                                                                                                                                                                                                                                                                                         \
            exit(EXIT_FAILURE);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
        }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \
    } while(0)

// CUDA kernel to extract a specific channel from a 4-channel array
__global__ void extractChannelKernel(unsigned char *input, unsigned char *output, int width, int height, int channelIdx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height) {
        int idx = y * width + x;
        int inputIdx = idx * 4 + channelIdx; // 4 channels (RGBA)
        output[idx] = input[inputIdx];
    }
}

int main() {

    const int width = 4;
    const int height = 4;
    const int numChannels = 4;

    // Host data for a 4-channel uchar4 array (RGBA)
    const int channelSize = width * height;
    const int totalSize = channelSize * numChannels;
    unsigned char h_data[totalSize];

    // Initialize data: simulate 4 channels (R, G, B, A)
    for(int i = 0; i < channelSize; i++) {
        h_data[i + 0 * channelSize] = (unsigned char)(i % 256);         // R channel
        h_data[i + 1 * channelSize] = (unsigned char)((i + 64) % 256);  // G channel
        h_data[i + 2 * channelSize] = (unsigned char)((i + 128) % 256); // B channel
        h_data[i + 3 * channelSize] = (unsigned char)((i + 192) % 256); // A channel
    }

    // Allocate CUDA array with 4-channel unsigned char format
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArrayDefault));

    // Copy data to CUDA array
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, h_data, width * numChannels, width * numChannels, height, cudaMemcpyHostToDevice));

    // Allocate linear device memory to copy array data
    unsigned char *d_linear;
    CUDA_CHECK(cudaMalloc(&d_linear, totalSize));

    // Copy array to linear memory
    CUDA_CHECK(cudaMemcpy2DFromArray(d_linear, width * numChannels, cuArray, 0, 0, width * numChannels, height, cudaMemcpyDeviceToDevice));

    // Allocate device memory for the extracted channel
    unsigned char *d_channel;
    CUDA_CHECK(cudaMalloc(&d_channel, channelSize));

    // Launch kernel to extract channel 0 (R)
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    extractChannelKernel<<<gridDim, blockDim>>>(d_linear, d_channel, width, height, 0);
    cudaError_t err = cudaGetLastError();
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy extracted channel back to host
    unsigned char h_channel[channelSize];
    CUDA_CHECK(cudaMemcpy(h_channel, d_channel, channelSize, cudaMemcpyDeviceToHost));

    // Print extracted channel data
    printf("Extracted Channel 0 (R) data:\n");
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            printf("%3u ", h_channel[i * width + j]);
        }
        printf("\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_channel));
    CUDA_CHECK(cudaFree(d_linear));
    CUDA_CHECK(cudaFreeArray(cuArray));

    return 0;
}
