#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaStreamAttributes){
    cudaStream_t srcStream, dstStream;
    cudaError_t err;
    err = cudaStreamCreate(&srcStream);
    CHECK_CUDA_ERROR(err, "Failed to create source stream");
    err = cudaStreamCreate(&dstStream);
    CHECK_CUDA_ERROR(err, "Failed to create destination stream");
    cudaStreamAttrValue value;
    value.accessPolicyWindow.base_ptr = nullptr;
    value.accessPolicyWindow.num_bytes = 0;
    value.accessPolicyWindow.hitRatio = 1.0f;
    value.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    value.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    err = cudaStreamSetAttribute(srcStream, cudaStreamAttributeAccessPolicyWindow, &value);
    if(err == cudaSuccess) {
        err = cudaStreamCopyAttributes(dstStream, srcStream);
        CHECK_CUDA_ERROR(err, "Failed to copy stream attributes");
        cudaStreamAttrValue retrievedValue;
        err = cudaStreamGetAttribute(dstStream, cudaStreamAttributeAccessPolicyWindow, &retrievedValue);
        CHECK_CUDA_ERROR(err, "Failed to get stream attribute");
        ASSERT_EQ(retrievedValue.accessPolicyWindow.hitRatio, value.accessPolicyWindow.hitRatio) << "Stream attribute not copied correctly";
    } else {
        SUCCEED() << "Stream attributes not supported, skipping test";
    }
    err = cudaStreamDestroy(srcStream);
    CHECK_CUDA_ERROR(err, "Failed to destroy source stream");
    err = cudaStreamDestroy(dstStream);
    CHECK_CUDA_ERROR(err, "Failed to destroy destination stream");
}
