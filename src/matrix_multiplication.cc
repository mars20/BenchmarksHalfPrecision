#include <cmath>
#include <cstdlib>
#include <iostream>
#include "stats.h"
#include "half.hpp"
#include "riscv_extensions.h"
#include "half_precision_util.h"
#include "neon_extensions.h"

using half_float::half;

typedef half float16;

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr,
            "<binary> <matrix_a_rows> <matrix_a_cols/matrix_b_rows> <matrix_b_cols>\n");
    return 0;
  }

  int matrix_a_rows, matrix_b_rows, matrix_b_cols;
  matrix_a_rows = atoi(argv[1]);
  matrix_b_rows = atoi(argv[2]); // same as matrix_a_cols
  matrix_b_cols = atoi(argv[3]);
  int matrix_a_cols = matrix_b_rows;

  // Allocate memory for the inputs and outputs

  float16 *matrix_a = (float16 *)malloc(matrix_a_rows * matrix_a_cols * sizeof(float16));
  float16 *matrix_b = (float16 *)malloc(matrix_b_rows * matrix_b_cols * sizeof(float16));
  float *matrix_a_full = (float *)malloc(matrix_a_rows * matrix_a_cols * sizeof(float));
  float *matrix_b_full = (float *)malloc(matrix_b_rows * matrix_b_cols * sizeof(float));
  float16 *result_scalar = (float16 *)malloc(matrix_a_rows * matrix_b_cols * sizeof(float16));
  float *result_scalar_full = (float *)malloc(matrix_a_rows * matrix_b_cols * sizeof(float));
  float16 *result_vect = (float16 *)malloc(matrix_a_rows * matrix_b_cols * sizeof(float16));
  

  printf("[info] Initialized Matrix A\n");
  initializeMatrix(matrix_a, matrix_a_rows * matrix_a_cols, true);

  printf("[info] Initialized Matrix B\n");  
  initializeMatrix(matrix_b, matrix_b_rows * matrix_b_cols, true);

  initializeMatrix(matrix_a_full, matrix_a_rows * matrix_a_cols, true);
  initializeMatrix(matrix_b_full, matrix_b_rows * matrix_b_cols, true);


  printf("[info] Scalar Matrix Matrix Multiplication\n");

  #ifdef PROF_RISCV
  riscv::stats::csr counters;
  riscv::stats::StartStats(&counters);  // enable csr counters
  #endif

  #ifdef ARM_GEM5
  m5_reset_stats(0,0);
  #endif

  ScalarMatrixMatrixMultiply(matrix_a, matrix_b, result_scalar,
                             matrix_a_rows, matrix_b_rows,
                             matrix_b_cols);

  #ifdef ARM_GEM5
  m5_dump_stats(0, 0);
  #endif
  
  #ifdef PROF_RISCV
  riscv::stats::StopStats(&counters);    // disable csr counters
  riscv::stats::PrintStats(&counters);
  #endif
  
  printf("[info] Scalar Matrix Matrix Multiplication with floating point\n");

  #ifdef PROF_RISCV
  riscv::stats::StartStats(&counters);  // enable csr counters
  #endif
  
  #ifdef ARM_GEM5
  m5_reset_stats(0,0);
  #endif

  ScalarMatrixMatrixMultiply(matrix_a_full, matrix_b_full, result_scalar_full,
  			     matrix_a_rows, matrix_b_rows,
  			     matrix_b_cols);

  #ifdef ARM_GEM5
  m5_dump_stats(0, 0);
  #endif

  printf("[info] Vector Matrix Matrix Multiplication\n");

  #ifdef VNEON

  #ifdef ARM_GEM5
  m5_reset_stats(0,0);
  #endif

  NeonMatrixMatrixMultiply(matrix_a_full, matrix_a_rows, matrix_b_rows,
                           matrix_b_full, matrix_b_cols, result_scalar_full);

  #ifdef ARM_GEM5
  m5_dump_stats(0, 0);
  #endif

  #endif

  #ifdef PROF_RISCV
  riscv::stats::StopStats(&counters);    // disable csr counters
  riscv::stats::PrintStats(&counters);
  #endif
  
  #ifdef PROF_RISCV
  riscv::stats::StartStats(&counters);  // enable csr counters
  #endif
  
  #ifdef VRISCV
  VectorMatrixMatrixMultiply(matrix_a, matrix_b, result_vect,
                             matrix_a_rows, matrix_b_rows,
                             matrix_b_cols);
  #endif
  
  #ifdef PROF_RISCV
  riscv::stats::StopStats(&counters);    // disable csr counters
  riscv::stats::PrintStats(&counters);
  #endif
  // printMatrix(matrix_a,  matrix_a_rows, matrix_b_rows);
  // printMatrix(result_scalar,  matrix_a_rows, matrix_b_cols);
  // printMatrix(result_vect,  matrix_a_rows, matrix_b_cols);

  //  for(int i=0;i<matrix_a_rows * matrix_b_cols; i++){
  //    float diff = result_scalar[i] - result_vect[i];
  //    if(diff >  1e-5 || diff <  -1e-5){
  //      printf("Result at index: %d differs by %f\n", i, diff);
  //      }
  //  }


  free(matrix_a);
  free(matrix_b);
  free(result_scalar);
  free(matrix_a_full);
  free(matrix_b_full);
  free(result_scalar_full);
  free(result_vect);

  return 0;
}
