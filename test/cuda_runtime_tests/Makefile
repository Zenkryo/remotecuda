# Makefile for CUDA Runtime API Unified Test Program
# Compiler and flags
NVCC = nvcc
CXX = g++

# CUDA and C++ flags
NVCC_FLAGS = -std=c++14 -O2 -arch=sm_75 -Xcompiler -fPIC
CXX_FLAGS = -std=c++14 -O2 -fPIC

# Include directories
INCLUDES = -I/usr/local/cuda/include

# Library directories and libraries
LIBS = -L/usr/local/cuda/lib64 -lcuda -lcudart -lgtest -lgtest_main -lpthread

# Source files
COMMON_SRC = common.cu
COMMON_OBJ = common.o

# Find all test source files (excluding common.cu)
TEST_SOURCES = $(filter-out common.cu, $(wildcard test_*.cu))
TEST_OBJECTS = $(TEST_SOURCES:.cu=.o)

# Main executable name
MAIN_EXECUTABLE = cuda_runtime_tests

# Default target
all: $(MAIN_EXECUTABLE)

# Rule to compile common.cu to object file
$(COMMON_OBJ): $(COMMON_SRC) common.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Rule to compile test source files to object files
%.o: %.cu common.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Rule to link all test objects into one executable
$(MAIN_EXECUTABLE): $(TEST_OBJECTS) $(COMMON_OBJ)
	$(NVCC) $(NVCC_FLAGS) $^ $(LIBS) -o $@

# Run all tests in the unified executable
test: $(MAIN_EXECUTABLE)
	@echo "Running unified CUDA runtime API tests..."
	./$(MAIN_EXECUTABLE)

# Run tests with verbose output
test-verbose: $(MAIN_EXECUTABLE)
	@echo "Running unified CUDA runtime API tests with verbose output..."
	./$(MAIN_EXECUTABLE) --gtest_output=xml:test_results.xml

# Run tests with specific filter
test-filter: $(MAIN_EXECUTABLE)
	@echo "Running filtered tests (use FILTER=pattern)..."
	./$(MAIN_EXECUTABLE) --gtest_filter=$(FILTER)

# Run the unified executable
run: $(MAIN_EXECUTABLE)
	@echo "Running unified test executable..."
	./$(MAIN_EXECUTABLE)

# Clean build artifacts
clean:
	rm -f *.o $(MAIN_EXECUTABLE) *.xml

# Clean and rebuild everything
rebuild: clean all

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y libgtest-dev cmake build-essential
	cd /usr/src/gtest && sudo cmake . && sudo make && sudo cp *.a /usr/lib

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || echo "NVCC not found"
	@nvidia-smi || echo "nvidia-smi not found"

# Show available tests
list-tests: $(MAIN_EXECUTABLE)
	@echo "Available tests in unified executable:"
	./$(MAIN_EXECUTABLE) --gtest_list_tests

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build unified test executable"
	@echo "  test         - Run all tests in unified executable"
	@echo "  test-verbose - Run all tests with verbose output"
	@echo "  test-filter  - Run filtered tests (use FILTER=pattern)"
	@echo "  run          - Run the unified executable"
	@echo "  clean        - Remove build artifacts"
	@echo "  rebuild      - Clean and rebuild everything"
	@echo "  install-deps - Install required dependencies"
	@echo "  check-cuda   - Check CUDA installation"
	@echo "  list-tests   - Show available tests in executable"
	@echo "  help         - Show this help message"

# Phony targets
.PHONY: all test test-verbose test-filter run clean rebuild install-deps check-cuda list-tests help

# Debug information
debug:
	@echo "NVCC: $(NVCC)"
	@echo "NVCC_FLAGS: $(NVCC_FLAGS)"
	@echo "INCLUDES: $(INCLUDES)"
	@echo "LIBS: $(LIBS)"
	@echo "TEST_SOURCES: $(TEST_SOURCES)"
	@echo "TEST_OBJECTS: $(TEST_OBJECTS)"
	@echo "MAIN_EXECUTABLE: $(MAIN_EXECUTABLE)"
