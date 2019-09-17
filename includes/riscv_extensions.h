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
#ifndef RISCV_EXTENSIONS_H_
#define RISCV_EXTENSIONS_H_

#include "half.hpp"

#ifdef VRISCV

using half_float::half;

typedef half float16;

enum vconfig {
  kElementWidthMax8 = 0x120,
  kElementWidthMax16 = 0x220,
  kElementWidthMax32 = 0x320,
  kElementWidthMax64 = 0x420
};  // element width and number of registers

enum maxvlen {
  kMaxVectorLength8 = 32,
  kMaxVectorLength16 = 16,
  kMaxVectorLength32 = 8,
  kMaxVectorLength64 = 4
}; // maximum vector length


inline void SetVcfg(unsigned int config) {
  asm("csrw vcfg, %0\t\n" : : "r"(config));
}

inline void SetVl(unsigned int len) {
  asm("csrw vl, %0\t\n" : : "r"(len));
}

inline void SetConfig(unsigned int maxew,
                      unsigned int maxvl) {
  asm("csrw vcfg, %0 \t\n"
      "csrw vl, %1 \t\n"
      :
      : "r"(maxew), "r"(maxvl));
}

template <class T>
inline void __VectorLoadInput1(const T* load_address) {
  asm volatile("vlsd va1, 0(%0), v \t\n"
               :
               : "r"(load_address));
}

template <class T>
inline void __VectorLoadInput1Scalar(const T* load_address) {
  asm volatile( "vlsd va1, 0(%0), s \t\n"
               :
               : "r"(load_address));
}

template <class T>
inline void __VectorLoadInput2(const T* load_address) {
  asm volatile("vlsd va2, 0(%0), v \t\n"
               :
               : "r"(load_address));
}

template <class T>
inline void __VectorLoadInput1(const T* load_address, int stride) {
  asm volatile("vlsd va1, 0(%0), %1, v \t\n"
               :
               : "r"(load_address), "r"(stride));
}

template <class T>
inline void __VectorLoadInput2(const T* load_address, int stride) {
  asm volatile("vlsd va2, 0(%0), %1, v \t\n"
               :
               : "r"(load_address), "r"(stride));
}

template <class T>
inline void __VectorLoadActivationInput(T* load_address) {
  asm volatile("vlsd vt11, 0(%0), v\t\n"
               :
               : "r"(load_address));
}

template <class T>
inline void __VectorLoadActivationInput(T* load_address, int stride) {
  asm volatile("vlsd vt11, 0(%0), %1, v \t\n"
               :
               : "r"(load_address), "r"(stride));
}

template <class T>
inline void __VectorLoadBias(const T* load_address) {
  asm volatile("vlsd vt4, 0(%0), v \t\n"
               :
               : "r"(load_address));
}

template <class T>
inline void __VectorLoadBias(const T* load_address, int stride) {
  asm volatile("vlsd vt4, 0(%0), %1, v \t\n"
               :
               : "r"(load_address), "r"(stride));
}

template <class T>
inline void __VectorLoadPartialOutput(T* load_address) {
  asm volatile("vlsd vt4, 0(%0), v \t\n"
               :
               : "r"(load_address));
}

template <class T>
inline void __VectorLoad(const T* load_address1,
                         const T* load_address2) {
  asm volatile(
      "vlsd va1, 0(%0), v \t\n"
      "vlsd va2, 0(%1), v \t\n"
      :
      : "r"(load_address1), "r"(load_address2));
}

template <class T>
inline void __VectorLoad(const T* load_address1, const T* load_address2,
                         int stride) {
  asm volatile(
               
               "vlsd va1, 0(%0), v \t\n"
               "vlsd va2, 0(%1), %2, v \t\n"
               :
               : "r"(load_address1), "r"(load_address2), "r"(stride));
}

template <class T>
inline void __VectorLoadPartialOutput(T* load_address, int stride) {
  asm volatile(
               "vlsd vt4, 0(%0), %1, v \t\n"
               :
               : "r"(load_address), "r"(stride));
}

template <class T>
inline void __VectorBroadcastAccum(T accum) {
  asm volatile("vbcastx vt4, %0 \t\n" : : "r"(accum));
}

inline void __VectorClearAccum() {
  asm volatile("vbcastx vt4, zero \t\n");
}

inline void __VectorBroadcastInput(int input_index) {
  asm volatile("vsplat va1, vt2, %0 \t\n" : : "r"(input_index));
}

template <class T>
inline void __VectorBroadcastMinMax(T value_minimum, T value_maximum) {
  asm volatile(
      "vbcastx vt2, %0 \t\n"
      "vbcastx vt3, %1 \t\n"
      :
      : "r"(value_minimum), "r"(value_maximum));
}

inline void __VectorReduceAccumFloat() {
  asm volatile("vfredsum vt11, vt4 \t\n");
}

inline void __VectorAddFloat() {
  asm volatile("vfadd vt11, va1, va2, v \t\n");
}

inline void __VectorMulFloat() {
  asm volatile("vfmul vt11, va1, va2, v \t\n");
}

inline void __VectorMulAccFloat() {
  asm volatile("vfmadd vt4, va1, va2, vt4, v \t\n");
}

inline void __VectorAccFloat() {
  asm volatile("vfadd vt4, va1, vt4, v \t\n");
}

inline void __VectorMinMaxFloat() {
  asm volatile(
      "vfmax vt11, vt11, vt2 \t\n"
      "vfmin vt11, vt11, vt3 \t\n");
}

inline void __VectorMergeFloat() {
  asm volatile("vmerge vt11, vt11, vt10, t \t\n");
}
inline void __VectorSetMask(unsigned int idx, unsigned int val) {
  asm volatile("vinsx vt1, %1, %0, v \t\n" : : "r"(idx), "r"(val));
}

inline void __VectorResetMask(unsigned int idx) {
  asm volatile("vinsx vt1, zero, %0, v \t\n" : : "r"(idx));
}

inline void __VectorResetMaskAll() {
  asm volatile("vbcastx vt1, zero \t\n");
}

template <class T>
inline void __VectorStore(T* store_address) {
  asm volatile(
               "vssd vt11, 0(%0), v \t\n"
               :
               : "r"(store_address));
}

template <class T>
inline void __VectorStorePartialOutput(T* store_address, int stride) {
  asm volatile(
               "vssd vt4, 0(%0), %1, v \t\n"
               :
               : "r"(store_address), "r"(stride));
}

template <class T>
inline void __VectorStorePartialOutput(T* store_address) {
  asm volatile(
               "vssd vt4, 0(%0), v \t\n"
               :
               : "r"(store_address));
}

template <class T>
inline void __VectorStoreAccum(T* store_address) {
  // printf("in store accum, storing at, %f, %x\n",store_address, store_address);
  asm volatile(
               "vssd vt11, 0(%0), s \t\n"
               :
               : "r"(store_address));
}

inline void __VectorSplatMulAccFloat(int idx) {
  asm volatile("vsplat vs3, va1, %0, v \t\n"
               "vfmadd vt4, vs3, va2, vt4, v \t\n"
               :
               : "r"(idx)
               );
}

//inline void MatrixMatrixMultiplyBlocked(float16 *matrix_a,
//                                        float16 *matrix_b,
//                                        float16 *result,
//                                        int matrix_a_rows,
//                                        int matrix_b_rows, // matrix_a_cols
//                                        int matrix_b_cols) {

  // c[m][n]= a[m][k]*b[k][n]

//  int new_matrix_b_rows = matrix_b_rows -
//                          (matrix_b_rows & (kMaxVectorLength16 - 1));
//  int matrix_b_rows_diff = matrix_b_rows & (kMaxVectorLength16 - 1);

//  SetVcfg(kElementWidthMax16);

//  for (int r_a = 0; r_a < matrix_a_rows; r_a++) {
//    for (int c_b = 0; r_a < matrix_b_rows; c_b++) { 
//      SetVl(kMaxVectorLength16);
//      __VectorClearAccum();
//      for (int r_b = 0; r_b < matrix_b_rows; r_b++) {
//        __VectorLoadInput1Scalar(matrix_a + r_a * matrix_b_rows + r_b);
//        __VectorLoadInput2(matrix_b + r_b * matrix_b_cols + c_b);
//        __VectorMulAccFloat();
        // printf("r_a:%d, c_b:%d, r_b(c_a):%d\n", r_a, c_b, r_b);
//      }
//      __VectorStorePartialOutput(result + r_a * matrix_b_cols + c_b);
//    }
//    if (matrix_b_cols_diff != 0) {
//      SetVl(matrix_b_cols_diff);
//      __VectorClearAccum();
//      for (int r_b = 0; r_b < matrix_b_rows; r_b++) {
//        __VectorLoadInput1Scalar(matrix_a + r_a * matrix_b_rows + r_b);
//        __VectorLoadInput2(matrix_b + r_b * matrix_b_cols + new_matrix_b_cols);
//        __VectorMulAccFloat();
//        // printf("r_a:%d, matrix_b_cols_diff:%d, r_b(c_a):%d\n",
//        // r_a, matrix_b_cols_diff, r_b);
//      }
//      __VectorStorePartialOutput(result + r_a * matrix_b_cols + new_matrix_b_cols);
//    }
//  }
//}

inline void VectorMatrixMatrixMultiply(float16 *matrix_a,
                                       float16 *matrix_b,
                                       float16 *result,
                                       int matrix_a_rows,
                                       int matrix_b_rows, // matrix_a_cols
                                       int matrix_b_cols) {

  // c[m][n]= a[m][k]*b[k][n]

    int new_matrix_b_cols = matrix_b_cols -
      (matrix_b_cols & (kMaxVectorLength16 - 1));
    int matrix_b_cols_diff = matrix_b_cols & (kMaxVectorLength16 - 1);

    SetVcfg(kElementWidthMax16);

    for (int r_a = 0; r_a < matrix_a_rows; r_a++) {
      for (int c_b = 0; c_b < new_matrix_b_cols; c_b += kMaxVectorLength16) {
        SetVl(kMaxVectorLength16);
        __VectorClearAccum();
        for (int r_b = 0; r_b < matrix_b_rows; r_b++) {
          __VectorLoadInput1Scalar(matrix_a + r_a * matrix_b_rows + r_b);
          __VectorLoadInput2(matrix_b + r_b * matrix_b_cols + c_b);
          __VectorMulAccFloat();
          // printf("r_a:%d, c_b:%d, r_b(c_a):%d\n", r_a, c_b, r_b);
        }
        __VectorStorePartialOutput(result + r_a * matrix_b_cols + c_b);
      }
      if (matrix_b_cols_diff != 0) {
        SetVl(matrix_b_cols_diff);
        __VectorClearAccum();
        for (int r_b = 0; r_b < matrix_b_rows; r_b++) {
          __VectorLoadInput1Scalar(matrix_a + r_a * matrix_b_rows + r_b);
          __VectorLoadInput2(matrix_b + r_b * matrix_b_cols + new_matrix_b_cols);
          __VectorMulAccFloat();
          // printf("r_a:%d, matrix_b_cols_diff:%d, r_b(c_a):%d\n",
          // r_a, matrix_b_cols_diff, r_b);
        }
        __VectorStorePartialOutput(result + r_a * matrix_b_cols + new_matrix_b_cols);
      }
    }
}
inline void VectorSparseMatrixVectorMultiplication(float16* sp_data,
                                                   float16* vect,
                                                   float16* result,
                                                   int matrix_rows,
                                                   int* row_length,
                                                   int* col_idx){
  SetVcfg(kElementWidthMax16);

  float16 *sp_mat_ptr = sp_data;
  int *col_idx_ptr = col_idx;
  for(int i=0; i < matrix_rows; i++) {
    int postamble_start = row_length[i] -
                          (row_length[i] &(kMaxVectorLength16 - 1));
    SetVl(kMaxVectorLength16);
    asm(
        "vlsd va1, 0(%0), s \t\n"
        :
        :"r" (vect + i)
        );

    for(int j=0; j < postamble_start; j+=kMaxVectorLength16){
      asm(
          "vlsd va2, 0(%0), v \t\n"
          "vlsd va3, 0(%1), v \t\n"
          "vlxd va4, 0(%2), va3, v \t\n"
          "vfmadd va4, va1, va2, va4, v \t\n"
          "vsxd va4, (%2), va3, v \t\n"
          :
          :"r"(sp_mat_ptr), "r"(col_idx_ptr), "r"(result));

        // printf("i=%d, j=%d\n", i,j);

      // printf("result address 0x%x\n",result);
      //      for(int idx=0; idx < kMaxVectorLength16; idx++){
      //	printf("%d, %x\n", col_idx_ptr[idx], result+col_idx_ptr[idx]);
      // }
      sp_mat_ptr = sp_mat_ptr + kMaxVectorLength16;
      col_idx_ptr = col_idx_ptr + kMaxVectorLength16;
    }
    int len_diff = row_length[i] & (kMaxVectorLength16 - 1);
    if(len_diff) {
      SetVl(len_diff);
      asm(
          "vlsd va2, 0(%0), v \t\n"
          "vlsd va3, 0(%1), v \t\n"
          "vlxd va4, 0(%2), va3, v \t\n"
          "vfmadd va4, va1, va2, va4, v \t\n"
          "vsxd va4, (%2), va3, v \t\n"
          :
          :"r"(sp_mat_ptr), "r"(col_idx_ptr), "r"(result));

      // printf("i=%d, len_diff:%d\n", i, len_diff);

      //      printf("result address 0x%x\n",result);
      
      //      for(int idx=0; idx < len_diff; idx++){
	// printf("%d\n", col_idx_ptr[idx]);
      //	printf("%d, %x\n", col_idx_ptr[idx], result+col_idx_ptr[idx]);
      //      }
      sp_mat_ptr = sp_mat_ptr + len_diff;
      col_idx_ptr = col_idx_ptr + len_diff;
    }
  }
}

inline void VectorMatrixVectorMultiplication(float16* matrix,
                                             float16* vector,
                                             float16* result,
                                             int matrix_rows,
                                             int matrix_cols) {
  // Vector length is equal to # columns
  // Output length is equal to # rows

  int new_cols = matrix_cols - (matrix_cols & (kMaxVectorLength16 - 1));
  int col_diff = matrix_cols & (kMaxVectorLength16 - 1);

  SetVcfg(kElementWidthMax16);

  SetVl(kMaxVectorLength16);

  for (int r = 0; r < matrix_rows; r++) {
    asm(
        "vlsd va1, 0(%0), s \t\n"
        :
        :"r" (vector + r)
        );
    for (int c = 0; c < new_cols; c += kMaxVectorLength16) {
      asm(
          "vlsd va2, 0(%0), v \t\n"
          "vlsd va4, 0(%1), v \t\n"
          "vfmadd va4, va1, va2, va4, v \t\n"
          "vssd va4, 0(%1), v \t\n"
          :
          :"r"(matrix+ r*matrix_cols + c), "r"(result + c));

      // printf("r=%d, c=%d\n", r, c);
    }

    if (col_diff != 0) {
      SetVl(col_diff);
      asm(
          "vlsd va2, 0(%0), v \t\n"
          "vlsd va4, 0(%1), v \t\n"
          "vfmadd va4, va1, va2, va4, v \t\n"
          "vssd va4, 0(%1), v \t\n"
          :
          :"r"(matrix+ r*matrix_cols + new_cols), "r"(result + new_cols));
      // printf("r=%d, col_diff=%d\n", r, col_diff);
    }
  }
}

inline void VectorVectorMultiplication(float16* vector1,
                                       float16* vector2,
                                       float16 &result,
                                       int vect_len) {
  // Vector length is equal to # columns
  // Output length is equal to # rows
  // printf("%.4f, %x \n",result,&result);
  int new_len = vect_len - (vect_len & (kMaxVectorLength16 - 1));
  int len_diff = vect_len & (kMaxVectorLength16 - 1);

  SetVcfg(kElementWidthMax16);

  SetVl(kMaxVectorLength16);
  __VectorClearAccum();

    for (int l = 0; l < new_len; l += kMaxVectorLength16) {
      asm("vlsd va1, 0(%0), v \t\n"
          "vlsd va2, 0(%1), v \t\n"
          "vfmadd vt4, va1, va2, vt4, v \t\n"
          :
          :"r"(vector1+ l), "r"(vector2+ l));
    }

    if (len_diff != 0) {
      SetVl(len_diff);
      asm("vlsd va1, 0(%0), v \t\n"
          "vlsd va2, 0(%1), v \t\n"
          :
          :"r"(vector1 + new_len), "r"(vector2 + new_len));

      SetVl(kMaxVectorLength16);
      asm("vfmadd vt4, va1, va2, vt4, v  \t\n");
    }
    SetVl(kMaxVectorLength16);
    asm volatile("vfredsum vt11, vt4 \t\n");  // Reduce partial sum to single value
    // printf("gonna store\n");
    __VectorStoreAccum(&result);  // Store as scalar value
}
#endif 

#endif  // RISCV_EXTENSIONS_H_
