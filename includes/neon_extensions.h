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
#ifndef NEON_EXTENSIONS_H_
#define NEON_EXTENSIONS_H_

#include "half.hpp"

#ifdef VNEON

#include<arm_neon.h>
using half_float::half;

typedef half float16;

#define kFloatWeightsPerNeonLane 4

void NeonMatrixMatrixMultiply(float* matrix_a, int matrix_a_rows,
                              int matrix_b_rows, float* matrix_b,
                              int matrix_b_cols, float* result) {
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      matrix_b_cols - (matrix_b_cols & (kFloatWeightsPerNeonLane - 1));

    // const float* vector_in_batch = vector + b * m_cols;
    // const float* matrix_row = matrix;

    // Main matrix by vector multiplication loop
    for (int r_a = 0; r_a < matrix_a_rows; r_a++){
      for (int c_b = 0; c_b < postamble_start; c_b += kFloatWeightsPerNeonLane) {
        float32x4_t acc_32x4 = vmovq_n_f32(0.0);
        for (int r_b = 0; r_b < matrix_b_rows; r_b++) {
        // Load 4 float values from vector and matrix row.
        float32x4_t matrix_a_f32x4 = vld1q_dup_f32(matrix_a + r_a * matrix_b_rows + r_b);
        float32x4_t matrix_b_f32x4 = vld1q_f32(matrix_b + r_b * matrix_b_cols + c_b);
        // Multiply the vector and matrix row and add to accumulator.
        acc_32x4 = vmlaq_f32(acc_32x4, matrix_a_f32x4, matrix_b_f32x4);
        }
        vst1q_f32(result + r_a * matrix_b_cols + c_b, acc_32x4);
      }
      for (int c_b = postamble_start; c_b < matrix_b_cols; c_b++) {
        float partialProductSum = 0.0;
        for (int r_b = 0; r_b < matrix_b_rows; r_b++) {
          partialProductSum += matrix_a[r_b + r_a * matrix_b_rows]
                               * matrix_b[r_b * matrix_b_cols + c_b];
      }
      result[c_b + r_a * matrix_b_cols] = partialProductSum;
    }
  }
}


void NeonMatrixVectorMultiply(float* matrix, int m_rows,
                              int m_cols, float* vector,
                              float* result,
                              int result_stride) {
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
    const int postamble_start =
      m_cols - (m_cols & (kFloatWeightsPerNeonLane - 1));

    // float* result_in_batch = result + b * m_rows * result_stride;
    // const float* vector_in_batch = vector + b * m_cols;
    const float* matrix_row = matrix;

    // Main matrix by vector multiplication loop
    for (int r = 0; r < m_rows; r++) {
      float32x4_t acc_32x4 = vmovq_n_f32(0.0);
      for (int c = 0; c < postamble_start; c += kFloatWeightsPerNeonLane) {
        // Load 4 float values from vector and matrix row.
        float32x4_t vector_f32x4 = vld1q_f32(vector + c);
        float32x4_t matrix_f32x4 = vld1q_f32(matrix_row + c);
        // Multiply the vector and matrix row and add to accumulator.
        acc_32x4 = vmlaq_f32(acc_32x4, matrix_f32x4, vector_f32x4);
      }
      // Add the 4 intermediate sum values to get the final dot-prod value for
      // this column.
      *result +=
        (vgetq_lane_f32(acc_32x4, 0) + vgetq_lane_f32(acc_32x4, 1) +
         vgetq_lane_f32(acc_32x4, 2) + vgetq_lane_f32(acc_32x4, 3));
      for (int c = postamble_start; c < m_cols; c++) {
        *result += matrix_row[c] * vector[c];
      }
      matrix_row += m_cols;
      result += result_stride;
    }
}

void NeonVectorVectorCwiseProduct(const float* vector1, const float* vector2,
                                  int v_size, float* result) {
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
    const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));
    for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
      // Load 4 float values from vector1 and vector2.
      float32x4_t v1_f32x4 = vld1q_f32(vector1 + v);
      float32x4_t v2_f32x4 = vld1q_f32(vector2 + v);
      // Vector multiply 4 float
      float32x4_t mul_32x4 = vmulq_f32(v1_f32x4, v2_f32x4);
      // Save to result array.
      vst1q_f32(&result[v], mul_32x4);
    }
    for (int v = postamble_start; v < v_size; v++) {
      result[v] = vector1[v] * vector2[v];
    }
}


#endif // VNEON

#endif  // NEON_EXTENSIONS_H_
