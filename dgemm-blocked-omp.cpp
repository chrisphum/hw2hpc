#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "likwid-stuff.h"



const char* dgemm_desc = "Blocked dgemm, OpenMP-enabled";

void square_dgemm(int n, double* A, double* B, double* C) 
{
   for ( int i = 0; i < n; i++){

      // My old code, returned error for variable length array; not allowed for omp
      //   // Get Row A:
      //   int row_num = i;
      //   double (A_row)[n];
      //   for (int k=0; k <n; k++){
      //       A_row[k] = A[ (k * n) + row_num ];
      //   }

      for (int j = 0; j < n; j++) {
            

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

        }
    }
}

void getBlock( int n, int colStart, int rowStart, int block_size, double* block, double* matrix)
{
    int start = (colStart * n) + rowStart;
    int count = 0;
    for (int i = 0; i < block_size; i++)
    {
        for (int p = 0; p < block_size; p++){
            block[count] = matrix[start + (n * i) + p];
            count++;
        }
    }

}

void writeBlock( int n, int colStart, int rowStart, int block_size, double* block, double* matrix)
{
    int start = (colStart * n) + rowStart;
    int count = 0;
    for (int i = 0; i < block_size; i++)
    {
        for (int p = 0; p < block_size; p++){
            matrix[start + (n * i) + p] = block[count];
            count++;
        }
    }

}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
   int Nb = n / block_size;

   #pragma omp parallel for collapse(2)
   for (int i = 0; i < Nb; i++){
      for (int j = 0; j < Nb; j++){
         LIKWID_MARKER_START(MY_MARKER_REGION_NAME);
         double (C_block)[block_size*block_size];

         getBlock(n, j * block_size, i * block_size, block_size, C_block, C);

         for (int k = 0 ; k < Nb; k++) {
               double (A_block)[block_size*block_size];
               getBlock(n, k * block_size, i * block_size, block_size, A_block, A);

               double (B_block)[block_size*block_size];
               getBlock(n, j * block_size, k * block_size, block_size, B_block, B);

               square_dgemm(block_size, A_block, B_block, C_block);

         }

         writeBlock(n, j * block_size, i * block_size, block_size, C_block, C);
         LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
      }
   }
}
