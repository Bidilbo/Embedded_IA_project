#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mlp.h"

double relu(double x) 
{
    return x > 0 ? x : 0;
}

void multiply_matrices(int rowsA, int colsA, double **A, double *B, double * C) 
{
    for (int i = 0; i < colsA; i++) 
    {
        C[i] = 0;
    }

    for (int i = 0; i < rowsA; i++) 
    {
        for (int k = 0; k < colsA; k++) 
        {
            C[i] += A[i][k] * B[k];
        }

    }
}

/*void transpose(int row, int col, double matrix[row][col], double result[row][col])
{
    return 0;
}*/

void forward_pass(double *input, double **w1, double *b1, double **w2, double *b2, double *output, int input_size, int hidden_size, int output_size)
{

    double *hidden1 = (double*)malloc(hidden_size * sizeof(double));

    // Couche 1

    multiply_matrices(hidden_size, input_size, w1, input, hidden1);
    for(int h = 0; h < hidden_size; h++)
    {
        hidden1[h] += b1[h];
        hidden1[h] = relu(hidden1[h]);
    }

    // Couche 2

    multiply_matrices(output_size, hidden_size, w2, hidden1, output);
    for(int h = 0; h < output_size; h++)
    {
        output[h] += b2[h];
        output[h] = relu(output[h]);
        printf("%lf \n", output[h]);
    }

}

int argmax(double *array, int size) {
    int max_index = 0;
    for (int i = 1; i < size; i++) {
        if (array[i] > array[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}