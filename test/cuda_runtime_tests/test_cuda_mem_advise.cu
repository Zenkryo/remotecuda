#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaMemAdvise){
    SUCCEED() << "Skipping external memory test - current hardware does not support it";
}
