#include <cmath>
#include <cstdlib>
#include <iostream>
#include <malloc.h>
#include "stats.h"
#include "half.hpp"
#include "riscv_extensions.h"
#include "half_precision_util.h"

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
  float16 *result = (float16*) calloc(vect_len, sizeof(float16));

  //printf("[info] Initialized Vector A\n");
  initializeMatrix(vector1, vect_len, true);
  initializeMatrix(vector2, vect_len, true);
  
  printf("[info] Vector Matrix Matrix Multiplication\n");

  #ifdef PROF_RISCV
  riscv::stats::csr counters;
  riscv::stats::StartStats(&counters);  // enable csr counters
  #endif

  #ifdef VRISCV
  VectorVectorMultiplication(vector1, vector2, result, vect_len);
  #endif
  
  #ifdef PROF_RISCV
  riscv::stats::StopStats(&counters);    // disable csr counters

  riscv::stats::PrintStats(&counters);
  #endif
  
  free(vector1);
  free(vector2);
  free(result);
  return 0;
}
