#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaUserObject){
    cudaUserObject_t userObject;
    int data = 42;
    cudaError_t err = cudaUserObjectCreate(
        &userObject, &data,
        [](void *ptr) {
            int *data = static_cast<int *>(ptr);
            *data = 0;
        },
        1, cudaUserObjectNoDestructorSync);
    if(err == cudaSuccess) {
        err = cudaUserObjectRetain(userObject);
        CHECK_CUDA_ERROR(err, "Failed to retain user object");
        cudaGraph_t graph;
        err = cudaGraphCreate(&graph, 0);
        CHECK_CUDA_ERROR(err, "Failed to create graph");
        err = cudaGraphRetainUserObject(graph, userObject);
        CHECK_CUDA_ERROR(err, "Failed to retain user object in graph");
        err = cudaGraphReleaseUserObject(graph, userObject);
        CHECK_CUDA_ERROR(err, "Failed to release user object from graph");
        err = cudaUserObjectRelease(userObject);
        CHECK_CUDA_ERROR(err, "Failed to release user object");
        err = cudaGraphDestroy(graph);
        CHECK_CUDA_ERROR(err, "Failed to destroy graph");
    } else {
        SUCCEED() << "User objects not supported, skipping test";
    }
}
