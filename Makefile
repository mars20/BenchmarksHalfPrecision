MAKEFILE_DIR := ${CURDIR}

# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
	HOST_OS = windows
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		HOST_OS := linux

endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS := osx
	endif
endif

HOST_ARCH := $(shell if [[ $(shell uname -m) =~ i[345678]86 ]]; then echo x86_32; else echo $(shell uname -m); fi)

TARGET := $(HOST_OS)
TARGET_ARCH := $(HOST_ARCH)

CXXFLAGS := -O3 --std=c++11 #-march=rv64gv
CFLAGS := ${CXXFLAGS}
CFLAGS :=
ARFLAGS := -r
TARGET_TOOLCHAIN_PREFIX :=
C_PREFIX :=
#CXX := riscv64-unknown-elf-g++
#CC := riscv64-unknown-elf-gcc

LDFLAGS := -lm

INCLUDES := \
-I. \
-I $(MAKEFILE_DIR)/includes/
include $(wildcard $(MAKEFILE_DIR)/targets/*_makefile.inc)

CXX := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}g++
CC := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}gcc
AR := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}ar

#Benchmark sources
BENCHMARK_SRCS := \
$(wildcard $(MAKEFILE_DIR)/src/*.cc)

#GEM5 sources
GEM5_SRCS := \
$(MAKEFILE_DIR)/gem5_files/m5op_arm_A64.S

GENDIR := $(MAKEFILE_DIR)/gen/$(TARGET)_$(TARGET_ARCH)/
OBJDIR := $(GENDIR)obj/
BINDIR := $(GENDIR)bin/


BENCHMARK_BINARY := $(addprefix $(BINDIR), \
$(patsubst %.cc,%,$(patsubst %.c,%,$(BENCHMARK_SRCS))))

BENCHMARK_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(BENCHMARK_SRCS))))

GEM5_OBJ := $(addprefix $(OBJDIR), \
$(patsubst %.S,%.o, $(GEM5_SRCS)))

$(OBJDIR)%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)%.o: %.S
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

$(BENCHMARK_BINARY): $(BINDIR)% : $(OBJDIR)%.o $(GEM5_OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $@ $^ \
	$(LDFLAGS)

benchmark: $(BENCHMARK_BINARY)

all: $(BENCHMARK_BINARY)

.PHONY: clean

clean:
	rm -rf $(MAKEFILE_DIR)/gen

cleantarget:
	rm -rf $(OBJDIR)
	rm -rf $(BINDIR)
