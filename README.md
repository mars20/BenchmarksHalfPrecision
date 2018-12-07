# Benchmarks Half Precision

This repository hosts few benchmarks for RISC-V vector toolchain.

#### Prerequiste
Install the riscv-tools from: https://github.com/mars20/riscv-tools/tree/v-ext

#### Build from repository

```shell
$ git clone https://github.com/mars20/BenchmarksHalfPrecision.git
$ cd BenchmarksHalfPrecision
$ make benchmark
```

#### Execute a model

```shell
$ spike pk gen/bin/<path to binary>
```
