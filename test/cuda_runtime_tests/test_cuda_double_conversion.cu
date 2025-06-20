#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaDoubleConversion){
    double hostValue = 1.0;
    double deviceValue = hostValue;
    cudaSetDoubleForDevice(&deviceValue);
    cudaSetDoubleForHost(&deviceValue);
    ASSERT_DOUBLE_EQ(deviceValue, hostValue) << "Double conversion failed";
}
