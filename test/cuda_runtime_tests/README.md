# CUDA Runtime API Individual Tests

This directory contains individual test files for CUDA Runtime API functions, each testing specific functionality in isolation.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (version 8.0 or later)
- Google Test (gtest) library
- GCC/G++ compiler

## Installation

### Install Dependencies (Ubuntu/Debian)
```bash
make install-deps
```

### Check CUDA Installation
```bash
make check-cuda
```

## Building Tests

### Build All Tests
```bash
make all
```

### Build a Specific Test
```bash
make test_cuda_version
```

### Clean Build Artifacts
```bash
make clean
```

### Rebuild Everything
```bash
make rebuild
```

## Running Tests

### Run All Tests
```bash
make test
```

### Run All Tests with Verbose Output
```bash
make test-verbose
```

### Run a Specific Test
```bash
make run-test_cuda_version
```

### List Available Tests
```bash
make list-tests
```

## Test Structure

Each test file follows this structure:
- `common.h` - Common header with test base class and utilities
- `common.cu` - Common CUDA kernels and helper functions
- `test_*.cu` - Individual test files for specific CUDA functions

## Available Test Categories

The tests cover various CUDA Runtime API categories:

### Device Management
- `test_cuda_device_synchronize.cu`
- `test_cuda_device_reset.cu`
- `test_cuda_choose_device.cu`
- `test_cuda_set_valid_devices.cu`

### Memory Management
- `test_cuda_malloc_async.cu`
- `test_cuda_memcpy_async.cu`
- `test_cuda_memset.cu`
- `test_cuda_mem_get_info.cu`

### Stream Management
- `test_cuda_stream_destroy.cu`
- `test_cuda_stream_synchronize.cu`
- `test_cuda_stream_wait_event.cu`

### Event Management
- `test_cuda_event_create.cu`
- `test_cuda_event_record.cu`
- `test_cuda_event_synchronize.cu`

### Graph API
- `test_cuda_graph_operations.cu`
- `test_cuda_graph_node_operations.cu`
- `test_cuda_graph_clone_and_debug.cu`

### And many more...

## Makefile Targets

| Target | Description |
|--------|-------------|
| `all` | Build all test executables |
| `test` | Run all tests |
| `test-verbose` | Run all tests with verbose output |
| `run-<test>` | Run a specific test |
| `clean` | Remove build artifacts |
| `rebuild` | Clean and rebuild everything |
| `install-deps` | Install required dependencies |
| `check-cuda` | Check CUDA installation |
| `list-tests` | Show available tests |
| `help` | Show help message |
| `debug` | Show debug information |

## Customization

You can modify the Makefile to:
- Change CUDA architecture (`-arch=sm_XX`)
- Adjust optimization flags
- Add custom include paths
- Modify library paths

## Troubleshooting

### Common Issues

1. **NVCC not found**: Ensure CUDA toolkit is installed and in PATH
2. **gtest not found**: Install Google Test library
3. **Permission denied**: Check file permissions and CUDA device access
4. **Compilation errors**: Verify CUDA architecture compatibility

### Debug Information
```bash
make debug
```

This will show the current configuration and detected files.

## Contributing

When adding new tests:
1. Follow the naming convention: `test_<function_name>.cu`
2. Include the common header: `#include "common.h"`
3. Use the test base class: `TEST_F(CudaRuntimeApiTest, TestName)`
4. Add proper error checking with `CHECK_CUDA_ERROR` macro
