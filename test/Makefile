# 定义编译器和编译选项
NVCC = nvcc
NVCC_FLAGS = -arch=sm_35 -Wno-deprecated-gpu-targets -cudart shared -lcuda -lcublas -lpthread -I/usr/local/cuda-11.4/samples/common/inc -lgtest -lgtest_main

# 输出目录
BUILD_DIR = build

# 查找所有包含 .cu 文件的目录（只保留至少有一个 .cu 文件的目录）
CU_DIRS = $(shell find . -name "*.cu" -exec dirname {} \; | sort | uniq)
$(info CU_DIRS = $(CU_DIRS))

# 为每个目录生成目标名称（只保留最后一层目录名）
TARGETS = $(foreach dir,$(CU_DIRS),$(notdir $(subst ./,,$(dir))))
$(info TARGETS = $(TARGETS))

# 默认目标
all: $(BUILD_DIR) $(addprefix $(BUILD_DIR)/,$(TARGETS))

# 创建 build 目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 为每个目录创建规则，忽略编译错误并输出到 build 目录
define make-rule
$(BUILD_DIR)/$(1): $(wildcard $(2)/*.cu)
	-$(NVCC) $(NVCC_FLAGS) $$^ -o $$@  # 输出到 build/$(1)
$(info Generated rule: $(BUILD_DIR)/$(1): $(wildcard $(2)/*.cu))
endef

# 生成所有规则（传递目录路径和目标名称）
$(foreach dir,$(CU_DIRS),$(eval $(call make-rule,$(notdir $(subst ./,,$(dir))),$(dir))))

# 调试目标
print-rules:
	@echo "CU_DIRS = $(CU_DIRS)"
	@echo "TARGETS = $(TARGETS)"
	@$(foreach dir,$(CU_DIRS),echo "Rule: $(BUILD_DIR)/$(notdir $(subst ./,,$(dir))): $(wildcard $(dir)/*.cu)";)

# 清理目标
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean print-rules $(BUILD_DIR)
