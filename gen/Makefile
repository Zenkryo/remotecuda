CXX = g++
CXXFLAGS = -std=c++11 -fPIC -Wno-deprecated-declarations -I../ -I/usr/local/cuda/include -I/usr/local/cudnn/include -DCUBLASAPI= #-DDEBUG
LDFLAGS = -ldl -lpthread -luuid

# Compilation flags for hook.so
HOOK_CXXFLAGS = $(CXXFLAGS)
HOOK_LDFLAGS = $(LDFLAGS) -shared

# Compilation flags for server
SERVER_CXXFLAGS = $(CXXFLAGS)
SERVER_LDFLAGS = $(LDFLAGS) -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcudart -lcuda -lnvidia-ml -lcublas -Wl,-rpath,/usr/local/cuda/lib64

# Source files for hook.so
HOOK_SRCS = \
    hook_hidden_api.cpp \
    hook_cuda.cpp \
    hook_nvml.cpp \
    hook_cuda_runtime_api.cpp \
    hook_cublas_api.cpp \
    hook_client.cpp \
    ../manual_hook.cpp \
    ../rpc/rpc_buffers.cpp\
    ../rpc/rpc_conn.cpp\
    ../rpc/rpc_client.cpp\
    ../client.cpp

# Source files for server
SERVER_SRCS = \
    handle_hidden_api.cpp \
    handle_cuda.cpp \
    handle_nvml.cpp \
    handle_cuda_runtime_api.cpp \
    handle_cublas_api.cpp \
    handle_server.cpp \
    ../manual_handle.cpp \
    ../rpc/rpc_buffers.cpp \
    ../rpc/rpc_conn.cpp \
    ../rpc/rpc_server.cpp \
    ../server.cpp

HOOK_OBJS = $(HOOK_SRCS:.cpp=.o)
SERVER_OBJS = $(SERVER_SRCS:.cpp=.o)

all: hook.so server

hook.so: $(HOOK_OBJS)
	$(CXX) $(HOOK_CXXFLAGS) -o $@ $^ $(HOOK_LDFLAGS)

server: $(SERVER_OBJS)
	$(CXX) $(SERVER_CXXFLAGS) -o $@ $^ $(SERVER_LDFLAGS)

clean:
	rm -f $(HOOK_OBJS) $(SERVER_OBJS) hook.so server

.PHONY: all clean
