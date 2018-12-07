MAKEFILE_DIR := ${CURDIR}

CXX := riscv64-unknown-elf-g++
CC := riscv64-unknown-elf-gcc
CXXFLAGS := -march=rv64gv -O3 -std=gnu++11
CCFLAGS := -march=rv64gv -O3

LDFLAGS := -lm

INCLUDES := \
-I. \
-I $(MAKEFILE_DIR)/includes/

#Benchmark sources
BENCHMARK_SRCS := \
$(wildcard $(MAKEFILE_DIR)/src/*.cc)

GENDIR := $(MAKEFILE_DIR)/gen/
OBJDIR := $(GENDIR)obj/
BINDIR := $(GENDIR)bin/


BENCHMARK_BINARY := $(addprefix $(BINDIR), \
$(patsubst %.cc,%,$(patsubst %.c,%,$(BENCHMARK_SRCS))))

BENCHMARK_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(BENCHMARK_SRCS))))

$(OBJDIR)%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

$(BENCHMARK_BINARY): $(BINDIR)% : $(OBJDIR)%.o
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $@ $^ \
	$(LDFLAGS)

benchmark: $(BENCHMARK_BINARY)

all: $(BENCHMARK_BINARY)

.PHONY: clean

clean:
	rm -rf $(MAKEFILE_DIR)/gen


