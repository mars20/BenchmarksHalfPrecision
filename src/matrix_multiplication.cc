#include <cmath>
#include <cstdlib>
#include <iostream>
#include "stats.h"
#include "half.hpp"
#include "riscv_extensions.h"
#include "half_precision_util.h"

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
  matrix_b_rows = atoi(argv[2]);
  matrix_b_cols = atoi(argv[3]);
  int matrix_a_cols = matrix_b_rows;

  // Allocate memory for the inputs and outputs

  float16 *matrix_a = (float16 *)malloc(matrix_a_rows * matrix_a_cols * sizeof(float16));
  float16 *matrix_b = (float16 *)malloc(matrix_b_rows * matrix_b_cols * sizeof(float16));
  float16 *result_scalar = (float16 *)malloc(matrix_a_rows * matrix_b_cols * sizeof(float16));
  float16 *result_vect = (float16 *)malloc(matrix_a_rows * matrix_b_cols * sizeof(float16));

  initializeMatrix(matrix_a, matrix_a_rows * matrix_a_cols, true);
  initializeMatrix(matrix_b, matrix_b_rows * matrix_b_cols, true);

  riscv::stats::csr counters;

  printf("Scalar Matrix Matrix Multiplication\n");
  riscv::stats::StartStats(&counters);  // enable csr counters
  ScalarMatrixMatrixMultiply(matrix_a, matrix_b, result_scalar,
                             matrix_a_rows, matrix_b_rows,
                             matrix_b_cols);

  riscv::stats::StopStats(&counters);    // disable csr counters
  riscv::stats::PrintStats(&counters);

  printf("Vector Matrix Matrix Multiplication\n");
  riscv::stats::StartStats(&counters);  // enable csr counters
  VectorMatrixMatrixMultiply(matrix_a, matrix_b, result_vect,
                             matrix_a_rows, matrix_b_rows,
                             matrix_b_cols);
  riscv::stats::StopStats(&counters);    // disable csr counters
  riscv::stats::PrintStats(&counters);

  //printMatrix(matrix_a,  matrix_a_rows, matrix_b_rows);
  // printMatrix(result_scalar,  matrix_a_rows, matrix_b_cols);
  // printMatrix(result_vect,  matrix_a_rows, matrix_b_cols);

  // for(int i=0;i<matrix_a_rows * matrix_b_cols; i++){
  //   float diff = result_scalar[i] - result_vect[i];
  //   if(diff >  1e-5 || diff <  -1e-5){
  //     printf("Result at index: %d differs by %f\n", i, diff);
  //     }
  // }


  free(matrix_a);
  free(matrix_b);
  free(result_scalar);
  free(result_vect);

  return 0;
}
