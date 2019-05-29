#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include "stats.h"
#include "riscv_extensions.h"
#include "half_precision_util.h"


using half_float::half;

typedef half float16;

int main(int argc, char** argv) {
  if(argc != 4){
    fprintf(stderr, "<binary> <matrix_rows> <matrix_cols> <sparsity rate [0.1-0.9]> \n");
    return 1;
  }

  int matrix_rows, matrix_cols;
  float sparse_rate;
  matrix_rows = atoi(argv[1]);
  matrix_cols = atoi(argv[2]);
  sparse_rate = atof(argv[3]);
  int flat_length = matrix_rows * matrix_cols;
  int nz_elements = int(flat_length*(1.0-sparse_rate));

  // allocate memory for sparse matrix and set it to zero
  float16 *sparse_matrix = (float16*) calloc(flat_length, sizeof(float16));

  float *sparse_matrix_full = (float*) calloc(flat_length, sizeof(float));

  int *row_len = (int*) malloc(matrix_rows*sizeof(int));
  int *row_indices = (int*) malloc(nz_elements*sizeof(int));
  int *column_indices = (int*) malloc(nz_elements*sizeof(int));
  float16 *data = (float16*) malloc(nz_elements*sizeof(float16));

  float16 *vect = (float16*) malloc(matrix_rows*sizeof(float16));

  float *data_full = (float*) malloc(nz_elements*sizeof(float));

  float *vect_full = (float*) malloc(matrix_rows*sizeof(float));

  float *product_scalar_full = (float*) calloc(matrix_cols, sizeof(float));

  float16 *product_vect = (float16*) calloc(matrix_cols, sizeof(float16));
  float16 *product_sp_vect = (float16*) calloc(matrix_cols, sizeof(float16));
  float16 *product_scalar = (float16*) calloc(matrix_cols, sizeof(float16));

  initializeMatrix(vect, matrix_rows, true);
  initializeSparseMatrix(sparse_matrix, nz_elements,
                         matrix_rows, matrix_cols);
  ConvertSparseMatrixCSR(sparse_matrix, matrix_rows,
                         matrix_cols, data,
                         row_indices, column_indices,
                         row_len);

  printf("[info] Scalar Matrix Vector Multiplication\n");

  #ifdef PROF_RISCV
  riscv::stats::csr counters;
  riscv::stats::StartStats(&counters);  // enable csr counters
  #endif

  #ifdef ARM_GEM5
  m5_reset_stats(0,0);
  #endif
  
  MatrixVectorMultiplication(matrix_rows, matrix_cols, sparse_matrix,
                             vect, product_scalar);

  #ifdef ARM_GEM5
  m5_dump_stats(0, 0);
  #endif
  
  #ifdef PROF_RISCV
  riscv::stats::StopStats(&counters);    // disable csr counters
  riscv::stats::PrintStats(&counters);
  #endif

  printf("[info] Vector Matrix Vector Multiplication\n");

  #ifdef PROF_RISCV
  riscv::stats::StartStats(&counters);  // enable csr counters
  #endif

  #ifdef VRISCV
  VectorMatrixVectorMultiplication(sparse_matrix, vect, product_vect,
                                   matrix_rows, matrix_cols);
  #endif
  
  #ifdef PROF_RISCV
  riscv::stats::StopStats(&counters);    // disable csr counters
  riscv::stats::PrintStats(&counters);
  #endif

  printf("[info] Vector Sparse Matrix Vector Multiplication\n");

  #ifdef PROF_RISCV
  riscv::stats::StartStats(&counters);  // enable csr counters
  #endif

  #ifdef VRISCV
  VectorSparseMatrixVectorMultiplication(data, vect, product_sp_vect,
                                         matrix_rows, row_len, column_indices);
  #endif
  
  #ifdef PROF_RISCV
  riscv::stats::StopStats(&counters);    // disable csr counters
  riscv::stats::PrintStats(&counters);
  #endif
  
  initializeMatrix(vect_full, matrix_rows, true);
  
  initializeSparseMatrix(sparse_matrix_full, nz_elements,
			 matrix_rows, matrix_cols);

  ConvertSparseMatrixCSR(sparse_matrix_full, matrix_rows,
			 matrix_cols, data_full,
			 row_indices, column_indices,
			 row_len);
  printf("[info] Scalar Matrix Vector Multiplication fp32\n");

  #ifdef PROF_RISCV
  riscv::stats::StartStats(&counters);  // enable csr counters
  #endif

  #ifdef ARM_GEM5
  m5_reset_stats(0,0);
  #endif
  
  MatrixVectorMultiplication(matrix_rows, matrix_cols, sparse_matrix_full,
			     vect_full, product_scalar_full);

  #ifdef ARM_GEM5
  m5_dump_stats(0,0);
  #endif
  
  #ifdef PROF_RISCV
  riscv::stats::StopStats(&counters);    // disable csr counters
  riscv::stats::PrintStats(&counters);
  #endif

  free(vect);
  free(sparse_matrix);
  free(row_indices);
  free(vect_full);
  free(sparse_matrix_full);
  free(column_indices);
  free(data);
  free(data_full);
  free(row_len);
  free(product_vect);
  free(product_sp_vect);
  free(product_scalar);
  free(product_scalar_full);

  return 0;
}
