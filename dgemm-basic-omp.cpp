#include <iostream>
#include <omp.h>
#include "likwid-stuff.h"

const char* dgemm_desc = "Basic implementation, OpenMP-enabled, three-loop dgemm.";

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
void square_dgemm(int n, double* A, double* B, double* C) 
{
   #pragma omp parallel for collapse(2)
   for ( int i = 0; i < n; i++){

      // My old code, returned error for variable length array; not allowed for omp
      //   // Get Row A:
      //   int row_num = i;
      //   double (A_row)[n];
      //   for (int k=0; k <n; k++){
      //       A_row[k] = A[ (k * n) + row_num ];
      //   }

      for (int j = 0; j < n; j++) {
            
            LIKWID_MARKER_START(MY_MARKER_REGION_NAME);

            // My old code, returned error for variable length array; not allowed for omp
            // // Get Column B:
            // int column_num = j;
            // double (B_col)[n];
            // for (int k=0; k <n; k++){
            //     B_col[k] = B[ (column_num * n) + k];
            // }

            double holder = C[ (j*n) + i];
            for (int q = 0; q < n; q++){
               holder += A[(q*n) + i] * B[q + (j*n)];
            }
            C[ (j*n) + i] = holder;

            LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
        }
    }
}
