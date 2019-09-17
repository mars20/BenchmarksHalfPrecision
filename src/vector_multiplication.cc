#include <cmath>
#include <cstdlib>
#include <iostream>
#include <malloc.h>
#include "stats.h"
#include "half.hpp"
#include "riscv_extensions.h"
#include "half_precision_util.h"
#include "neon_extensions.h"

using half_float::half;
typedef half float16;


int main(int argc, char **argv) {

  if (argc != 2) {
    fprintf(stderr,"<binary> <vector_len>\n");
    return 1;
  }

  int vect_len;
  vect_len = atoi(argv[1]);

  //memalign(32, 256);
  //  float16 *vector1 = (float16*) memalign(64, vect_len*sizeof(float16));
  //  float16 *vector2 = (float16*) memalign(64, vect_len*sizeof(float16));
  //  float16 *result = (float16*) memalign(64, vect_len*sizeof(float16)); // 64 bytes (512 bits) aligned, vector_len * 16 bits of memory

  float16 *vector1 = (float16*) calloc(vect_len, sizeof(float16));
  float16 *vector2 = (float16*) calloc(vect_len, sizeof(float16));
  float16 result;
  float16 result_scalar;
  //float16 *result = (float16*) calloc(vect_len, sizeof(float16));
  //float16 *result_scalar = (float16*) calloc(vect_len, sizeof(float16));

  float *vector1_full = (float*) calloc(vect_len, sizeof(float));
  float *vector2_full = (float*) calloc(vect_len, sizeof(float));
  //  float *result_full = (float*) calloc(vect_len, sizeof(float));
  // float *result_full_scalar = (float*) calloc(vect_len, sizeof(float));
  float result_full;
  float result_full_scalar;
  
  //printf("[info] Initialized Vector A\n");
  initializeMatrix(vector1, vect_len, true);
  initializeMatrix(vector2, vect_len, true);

  initializeMatrix(vector1_full, vect_len, true);
  initializeMatrix(vector2_full, vect_len, true);

  printf("[info] Scalar Matrix Matrix Multiplication \n");

    #ifdef ARM_GEM5
  m5_reset_stats(0,0);
    #endif

    #ifdef PROF_RISCV
  riscv::stats::csr counters;
  riscv::stats::StartStats(&counters);  // enable csr counters
    #endif

  ScalarVectorVectorProduct(vector1, vector2, result_scalar, vect_len);

    #ifdef PROF_RISCV
  riscv::stats::StopStats(&counters);    // disable csr counters
  riscv::stats::PrintStats(&counters);
    #endif

    #ifdef ARM_GEM5
  m5_dump_stats(0, 0);
    #endif

  printf("[info] Scalar Matrix Matrix Multiplication floating point\n");

    #ifdef PROF_RISCV
  riscv::stats::StartStats(&counters);  // enable csr counters
    #endif

    #ifdef ARM_GEM5
  m5_reset_stats(0,0);
    #endif

  ScalarVectorVectorProduct(vector1_full, vector2_full, result_full_scalar, vect_len);

    #ifdef ARM_GEM5
  m5_dump_stats(0, 0);
    #endif

    #ifdef PROF_RISCV
  riscv::stats::StopStats(&counters);    // disable csr counters
  riscv::stats::PrintStats(&counters);
    #endif

  printf("[info] Vector Matrix Matrix Multiplication\n");

    #ifdef VNEON
    #ifdef ARM_GEM5
  m5_dump_stats(0, 0);
    #endif

  NeonVectorVectorProductAccumulate(vector1_full, vector2_full, vect_len, result_full);

    #ifdef ARM_GEM5
  m5_dump_stats(0, 0);
    #endif
    #endif

    #ifdef VRISCV
    #ifdef PROF_RISCV
  riscv::stats::StartStats(&counters);  // enable csr counters
    #endif

  VectorVectorMultiplication(vector1, vector2, result, vect_len);

    #ifdef PROF_RISCV
  riscv::stats::StopStats(&counters);    // disable csr counters

  riscv::stats::PrintStats(&counters);
    #endif
    #endif

  free(vector1);
  free(vector2);
  free(vector1_full);
  free(vector2_full);
  // free(result);
  // free(result_full);
  return 0;
}
