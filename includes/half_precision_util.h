/*Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef HALF_PRECISION_H_
#define HALF_PRECISION_H_

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "half.hpp"

using half_float::half;

typedef half float16;


template <class T>
void initializeMatrix(T *matrix, const int matrix_size, bool random = false) {
  for (int i = 0; i < matrix_size; i++) {
    float val = random ? ((static_cast<float>(rand()) / RAND_MAX)) : 0.0f;
    //float val = random ? 2.0f : 0.0f;
    matrix[i] = T(val);
  }
}

template <class T>
void initializeSparseMatrix(T *matrix, int non_zero_elements,
                            int matrix_rows, int matrix_cols) {
  // generate randomn sparse matrix
  // printf("Generated sparse matrix\n");
  for(int i = 0; i < non_zero_elements;) {
    int row_idx = rand() % matrix_rows;
    int col_idx = rand() % matrix_cols;

    if (matrix[row_idx*matrix_cols + col_idx]) {  // something already at this index
      continue;         // skip index
    }
    float val = static_cast<float>(rand()) / RAND_MAX;
    matrix[row_idx*matrix_cols + col_idx] = T(val);
    ++i;
  }
}

template <class T>
void ConvertSparseMatrixCSR(T *sparse_matrix, int matrix_rows,
                            int matrix_cols, T *data,
                            int* row_indices,
                            int* cols_indices,
                            int* row_len){
  int idx =0;
  for(int i=0; i< matrix_rows; i++) {
    int current_row_len =0;
    for(int j=0; j< matrix_cols; j++) {

      if (sparse_matrix[i*matrix_cols + j]) {
        row_indices[idx] = i;
        cols_indices[idx] = j;
        data[idx] = sparse_matrix[i*matrix_cols + j];
        ++idx;
        ++current_row_len;
      }
    }
   row_len[i] = current_row_len;
  }
}


template<class T>
void printMatrix(T *matrix, int rows, int cols) {
  printf("printing matrix\n");
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      float val = matrix[c + r*cols];
      printf("%.4f \t",val);
    }
    printf("\n");
  }
}


template <class T>
void ScalarMatrixMatrixMultiply(T *matrix_a,
                                T *matrix_b,
                                T *result,
                                int matrix_a_rows,
                                int matrix_b_rows, // matrix_a_cols
                                int matrix_b_cols) {

  T partialProductSum;

  for (int r_a = 0; r_a < matrix_a_rows; r_a++) {
    for (int c_b = 0; c_b < matrix_b_cols; c_b++) {
      partialProductSum = 0.0;
      for (int r_b = 0; r_b < matrix_b_rows; r_b++) {
        partialProductSum += matrix_a[r_b + r_a * matrix_b_rows]
                             * matrix_b[r_b * matrix_b_cols + c_b];
      }
      result[c_b + r_a * matrix_b_cols] = partialProductSum;
    }
  }
}

template<class T>
void MatrixVectorMultiplication(int matrix_rows,
                                int matrix_cols,
                                T* sp_mat,
                                T* vect,
                                T* result) {
  T partialProduct;

  for(int i=0; i<matrix_cols; i++){
    partialProduct = 0.0;
    for(int j=0; j<matrix_rows; j++){
      partialProduct += vect[j]*sp_mat[j*matrix_cols+i];
    }
    result[i] = partialProduct;
  }
}

template<class T>
void ScalarVectorVectorProduct(T* vector1,
                               T* vector2,
                               T* result,
                               int vector_length) {

  for(int i=0; i<vector_length; i++){
      result[i] += vector1[i]*vector2[i];
  }
}

#endif //HALF_PRECISION_H_
