# CUDA 函数及其持久内存参数

以下表格列出了所有 CUDA 函数中具有类似内存参数的函数，这些参数指向的内存需要在函数调用后保持有效，直到相关操作（如图形执行、流操作或内核执行）完成。表格明确了函数所属的 **API 或库**（如 CUDA Runtime API、CUDA Driver API、cuBLAS 等），并包括函数名、相关参数、参数描述和内存管理要求。基于 CUDA 12.x 版本。

| 函数名                              | 相关参数                              | 参数描述                                       | 内存管理要求                                       | 所属 API/库                |
|-------------------------------------|---------------------------------------|----------------------------------------------|--------------------------------------------------|---------------------------|
| `cudaGraphAddMemcpyNode`            | `pCopyParams`                         | 包含 `srcPtr` 和 `dstPtr` 的内存描述           | `pCopyParams` 可在调用后释放；`srcPtr.ptr` 和 `dstPtr.ptr` 需在图形执行完成前有效 | CUDA Runtime API          |
| `cudaGraphAddMemcpyNodeToSymbol`    | `src`                                 | 源内存指针（主机或设备）                       | `src` 需在图形执行完成前有效                     | CUDA Runtime API          |
| `cudaGraphAddMemcpyNodeFromSymbol`  | `dst`                                 | 目标内存指针（主机或设备）                     | `dst` 需在图形执行完成前有效                     | CUDA Runtime API          |
| `cudaGraphAddMemsetNode`            | `pMemsetParams`                       | 包含 `dst` 的设备内存指针                      | `pMemsetParams` 可在调用后释放；`dst` 需在图形执行完成前有效 | CUDA Runtime API          |
| `cudaGraphAddKernelNode`            | `pKernelParams`                       | 内核参数数组，包含内存指针                     | `pKernelParams` 可在调用后释放；参数中的内存指针需在图形执行完成前有效 | CUDA Runtime API          |
| `cudaGraphAddHostNode`              | `pHostParams`                         | 包含 `userData` 的用户数据指针                 | `pHostParams` 可在调用后释放；`userData` 需在主机回调执行完成前有效 | CUDA Runtime API          |
| `cudaGraphAddChildGraphNode`        | 子图形的内存参数（间接）              | 子图形中的内存参数                             | 子图形中的内存参数需在图形执行完成前有效         | CUDA Runtime API          |
| `cudaMemcpyAsync`                   | `dst`, `src`                          | 目标和源内存指针                               | `dst` 和 `src` 需在流操作完成前有效              | CUDA Runtime API          |
| `cudaMemcpyToSymbolAsync`           | `src`                                 | 源内存指针                                     | `src` 需在流操作完成前有效                       | CUDA Runtime API          |
| `cudaMemcpyFromSymbolAsync`         | `dst`                                 | 目标内存指针                                   | `dst` 需在流操作完成前有效                       | CUDA Runtime API          |
| `cudaMemsetAsync`                   | `devPtr`                              | 设备内存指针                                   | `devPtr` 需在流操作完成前有效                    | CUDA Runtime API          |
| `cudaMemcpy2DAsync`                 | `dst`, `src`                          | 目标和源内存指针（2D）                         | `dst` 和 `src` 需在流操作完成前有效              | CUDA Runtime API          |
| `cudaMemcpy3DAsync`                 | `p`                                   | `cudaMemcpy3DParms` 结构体，包含 `srcPtr` 和 `dstPtr` | `p` 可在调用后释放；`srcPtr.ptr` 和 `dstPtr.ptr` 需在流操作完成前有效 | CUDA Runtime API          |
| `cudaMemcpyPeerAsync`               | `dstDevice`, `srcDevice`              | 目标和源设备内存指针                           | `dstDevice` 和 `srcDevice` 需在流操作完成前有效  | CUDA Runtime API          |
| `cudaLaunchKernel`                  | `args`                                | 内核参数数组，包含内存指针                     | `args` 可在调用后释放；参数中的内存指针需在内核执行完成前有效 | CUDA Runtime API          |
| `cudaLaunchCooperativeKernel`       | `args`                                | 内核参数数组，包含内存指针                     | `args` 可在调用后释放；参数中的内存指针需在内核执行完成前有效 | CUDA Runtime API          |
| `cudaLaunchKernelEx`                | `args`                                | 内核参数数组，包含内存指针                     | `args` 可在调用后释放；参数中的内存指针需在内核执行完成前有效 | CUDA Runtime API          |
| `cudaHostRegister`                  | `hostPtr`                             | 主机内存指针                                   | `hostPtr` 需在注册期间及相关操作完成前有效       | CUDA Runtime API          |
| `cudaMallocAsync`                   | `devPtr`                              | 设备内存指针（分配后）                         | `devPtr` 需在异步分配完成前有效                  | CUDA Runtime API          |
| `cudaFreeAsync`                     | `devPtr`                              | 设备内存指针                                   | `devPtr` 需在异步释放完成前有效                  | CUDA Runtime API          |
| `cuGraphAddMemcpyNode`              | `pCopyParams`                         | 包含 `srcPtr` 和 `dstPtr` 的内存描述           | `pCopyParams` 可在调用后释放；`srcPtr` 和 `dstPtr` 需在图形执行完成前有效 | CUDA Driver API           |
| `cuGraphAddMemsetNode`              | `pMemsetParams`                       | 包含 `dst` 的设备内存指针                      | `pMemsetParams` 可在调用后释放；`dst` 需在图形执行完成前有效 | CUDA Driver API           |
| `cuGraphAddKernelNode`              | `kernelParams`                        | 内核参数数组，包含内存指针                     | `kernelParams` 可在调用后释放；参数中的内存指针需在图形执行完成前有效 | CUDA Driver API           |
| `cuGraphAddHostNode`                | `pHostParams`                         | 包含 `userData` 的用户数据指针                 | `pHostParams` 可在调用后释放；`userData` 需在主机回调执行完成前有效 | CUDA Driver API           |
| `cuGraphAddChildGraphNode`          | 子图形的内存参数（间接）              | 子图形中的内存参数                             | 子图形中的内存参数需在图形执行完成前有效         | CUDA Driver API           |
| `cuMemcpyAsync`                     | `dst`, `src`                          | 目标和源内存地址（CUdeviceptr）                | `dst` 和 `src` 需在流操作完成前有效              | CUDA Driver API           |
| `cuMemsetD8Async`                   | `dstDevice`                           | 设备内存地址（CUdeviceptr）                    | `dstDevice` 需在流操作完成前有效                 | CUDA Driver API           |
| `cuMemsetD16Async`                  | `dstDevice`                           | 设备内存地址（CUdeviceptr）                    | `dstDevice` 需在流操作完成前有效                 | CUDA Driver API           |
| `cuMemsetD32Async`                  | `dstDevice`                           | 设备内存地址（CUdeviceptr）                    | `dstDevice` 需在流操作完成前有效                 | CUDA Driver API           |
| `cuMemcpy2DAsync`                   | `pCopy`                               | `CUDA_MEMCPY2D` 结构体，包含 `src` 和 `dst`     | `pCopy` 可在调用后释放；`src` 和 `dst` 需在流操作完成前有效 | CUDA Driver API           |
| `cuMemcpy3DAsync`                   | `pCopy`                               | `CUDA_MEMCPY3D` 结构体，包含 `src` 和 `dst`     | `pCopy` 可在调用后释放；`src` 和 `dst` 需在流操作完成前有效 | CUDA Driver API           |
| `cuLaunchKernel`                    | `kernelParams`                        | 内核参数数组，包含内存指针                     | `kernelParams` 可在调用后释放；参数中的内存指针需在内核执行完成前有效 | CUDA Driver API           |
| `cuLaunchCooperativeKernel`         | `kernelParams`                        | 内核参数数组，包含内存指针                     | `kernelParams` 可在调用后释放；参数中的内存指针需在内核执行完成前有效 | CUDA Driver API           |
| `cuMemHostRegister`                 | `hostPtr`                             | 主机内存指针                                   | `hostPtr` 需在注册期间及相关操作完成前有效       | CUDA Driver API           |
| `cublasGemmEx`                      | `A`, `B`, `C`                         | 矩阵数据的设备内存指针                         | `A`, `B`, `C` 需在计算完成前有效                 | cuBLAS                    |
| `cublasSgemm`                       | `A`, `B`, `C`                         | 矩阵数据的设备内存指针                         | `A`, `B`, `C` 需在计算完成前有效                 | cuBLAS                    |
| `cublasDgemm`                       | `A`, `B`, `C`                         | 矩阵数据的设备内存指针                         | `A`, `B`, `C` 需在计算完成前有效                 | cuBLAS                    |
| `cublasGemmBatchedEx`               | `Aarray`, `Barray`, `Carray`          | 矩阵数组的设备内存指针                         | `Aarray`, `Barray`, `Carray` 需在计算完成前有效  | cuBLAS                    |
| `cufftExecC2C`                      | `idata`, `odata`                      | 输入和输出数据的设备内存指针                   | `idata`, `odata` 需在变换执行完成前有效          | cuFFT                     |
| `cufftExecR2C`                      | `idata`, `odata`                      | 输入和输出数据的设备内存指针                   | `idata`, `odata` 需在变换执行完成前有效          | cuFFT                     |
| `cufftPlan1d`                       | （间接）`in`/`out` 在执行时使用       | FFT 输入/输出数据的设备内存指针                | `in`, `out` 需在计划执行期间有效                 | cuFFT                     |
| `cufftPlan2d`                       | （间接）`in`/`out` 在执行时使用       | FFT 输入/输出数据的设备内存指针                | `in`, `out` 需在计划执行期间有效                 | cuFFT                     |
| `cusparseSpMV`                      | `x`, `y`                              | 稀疏向量和结果向量的设备内存指针               | `x`, `y` 需在计算完成前有效                      | cuSPARSE                  |
| `cusparseSpMM`                      | `A`, `B`, `C`                         | 稀疏矩阵和密集矩阵的设备内存指针               | `A`, `B`, `C` 需在计算完成前有效                 | cuSPARSE                  |

## 说明

1. **CUDA Runtime API**
   - 属于 CUDA Runtime API 的函数提供高层次接口，简化设备管理和内核执行。涉及的内存参数（如 `src`, `dst`, `args`）必须在异步操作（如流或图形执行）完成前保持有效。
   - 示例：`cudaGraphAddMemcpyNode` 的 `pCopyParams` 本身可在调用后释放，但 `srcPtr.ptr` 和 `dstPtr.ptr` 需在图形执行完成前有效。

2. **CUDA Driver API**
   - 属于 CUDA Driver API 的函数提供低层次控制，适合细粒度管理场景。参数通常使用 `CUdeviceptr` 或结构体（如 `CUDA_MEMCPY2D`），内存要求与 Runtime API 类似。
   - 示例：`cuMemcpyAsync` 的 `dst` 和 `src`（`CUdeviceptr` 类型）需在流操作完成前有效。

3. **cuBLAS**
   - cuBLAS 是一个线性代数库，涉及矩阵和向量操作。函数如 `cublasGemmEx` 的矩阵指针（`A`, `B`, `C`）必须在计算完成前保持有效。
   - 示例：`cublasSgemm` 的 `A`, `B`, `C` 是设备内存指针，需在矩阵乘法执行期间有效。

4. **cuFFT**
   - cuFFT 是快速傅里叶变换库，涉及输入/输出数据的设备内存。函数如 `cufftExecC2C` 的 `idata` 和 `odata` 需在变换 MSR
   - 示例：`cufftPlan1d` 创建的计划在执行时会访问输入/输出缓冲区，这些缓冲区需保持有效。

5. **cuSPARSE**
   - cuSPARSE 是稀疏矩阵运算库，涉及稀疏矩阵和向量的设备内存。函数如 `cusparseSpMV` 的 `x` 和 `y` 需在稀疏矩阵-向量乘法期间有效。

## 注意事项

- **内存生命周期**：所有列出函数的参数指向的内存必须在相关操作（如流执行、图形执行、内核执行或库计算）完成前保持有效，否则可能导致未定义行为或程序崩溃。
- **版本兼容性**：本表格基于 CUDA 12.x 版本，某些函数（如 `cudaMallocAsync`）在早期版本中可能不可用，请根据您的 CUDA SDK 版本确认。
- **间接参数**：对于 `cudaGraphAddChildGraphNode` 或 `cuGraphAddChildGraphNode`，内存参数可能嵌套在子图形中，需递归检查子图形的节点。
- **库依赖**：cuBLAS、cuFFT、cuSPARSE 等库需要单独链接，并在程序中初始化（如 `cublasCreate`, `cufftPlan1d`）。%
