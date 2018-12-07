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
void ScalarMatrixMatrixMultiply(T *matrix_a,
                                T *matrix_b,
                                T *result,
                                int matrix_a_rows,
                                int matrix_b_rows, // matrix_a_cols
                                int matrix_b_cols) {

  T partialProductSum;

  for (int r_a = 0; r_a < matrix_a_rows; r_a++) {
    for (int c_b = 0; c_b < matrix_b_cols; c_b++) {
      partialProductSum = T(0.0);
      for (int r_b = 0; r_b < matrix_b_rows; r_b++) {
        partialProductSum += matrix_a[r_b + r_a * matrix_b_rows]
                             * matrix_b[r_b * matrix_b_cols + c_b];
      }
      result[c_b + r_a * matrix_b_cols] = partialProductSum;
    }
  }
}


template <class T>
void initializeMatrix(T *matrix, const int matrix_size, bool random = false) {
  for (int i = 0; i < matrix_size; i++) {
    float val = random ? static_cast<float>(rand()) / RAND_MAX : 0.0f;
    matrix[i] = T(val);
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

#endif //HALF_PRECISION_H_
