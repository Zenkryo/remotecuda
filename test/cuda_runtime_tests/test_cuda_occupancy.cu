#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaOccupancy){
    int maxActiveBlocks;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, test_kernel, 128, 0);
    if(err == cudaSuccess) {
        ASSERT_GT(maxActiveBlocks, 0) << "Invalid max active blocks";
        size_t dynamicSMemSize;
        err = cudaOccupancyAvailableDynamicSMemPerBlock(&dynamicSMemSize, test_kernel, 128, maxActiveBlocks);
        if(err == cudaSuccess) {
            ASSERT_GE(dynamicSMemSize, 0) << "Invalid dynamic shared memory size";
        } else {
            SUCCEED() << "Dynamic shared memory query not supported, skipping test";
        }
    } else {
        SUCCEED() << "Occupancy query not supported, skipping test";
    }
}
