CXX = g++
CXXFLAGS = -std=c++11 -fPIC -Wno-deprecated-declarations -I/usr/local/cuda/include -I/usr/local/cudnn/include -DCUBLASAPI= -DDEBUG -DDUMP
LDFLAGS = -ldl -lpthread

# Compilation flags for hook.so
HOOK_CXXFLAGS = $(CXXFLAGS)
HOOK_LDFLAGS = $(LDFLAGS) -shared

# Compilation flags for server
SERVER_CXXFLAGS = $(CXXFLAGS)
SERVER_LDFLAGS = $(LDFLAGS) -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcudart -lcuda -lnvidia-ml -lcublas -Wl,-rpath,/usr/local/cuda/lib64

# Compilation flags for test
TEST_CXXFLAGS = $(CXXFLAGS)
TEST_LDFLAGS = $(LDFLAGS) -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcudart -lcuda -lnvidia-ml -lcublas -Wl,-rpath,/usr/local/cuda/lib64

# Source files for hook.so
HOOK_SRCS = \
    hook_hidden_api.cpp \
    hook_cuda.cpp \
    hook_nvml.cpp \
    hook_cuda_runtime_api.cpp \
    hook_cublas_api.cpp \
    hook_client.cpp \
    ../manual_hook.cpp \
    ../rpc.cpp\
    ../client.cpp

# Source files for server
SERVER_SRCS = \
    handle_hidden_api.cpp \
    handle_cuda.cpp \
    handle_nvml.cpp \
    handle_cuda_runtime_api.cpp \
    handle_cublas_api.cpp \
    handle_server.cpp\
    ../manual_handle.cpp\
    ../rpc.cpp\
    ../rpc_server.cpp\
    ../server.cpp

# Source files for test
TEST_SRCS = \
    ../test/test.cpp

HOOK_OBJS = $(HOOK_SRCS:.cpp=.o)
SERVER_OBJS = $(SERVER_SRCS:.cpp=.o)

TEST_OBJS = $(TEST_SRCS:.cpp=.o)

all: hook.so server test

hook.so: $(HOOK_OBJS)
	$(CXX) $(HOOK_CXXFLAGS) -o $@ $^ $(HOOK_LDFLAGS)

server: $(SERVER_OBJS)
	$(CXX) $(SERVER_CXXFLAGS) -o $@ $^ $(SERVER_LDFLAGS)

test: $(TEST_OBJS)
	$(CXX) $(TEST_CXXFLAGS) -o $@ $^ $(TEST_LDFLAGS)

clean:
	rm -f $(HOOK_OBJS) $(SERVER_OBJS) $(TEST_OBJS) hook.so server test

.PHONY: all clean
