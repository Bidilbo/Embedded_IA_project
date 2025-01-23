#include <math.h>
#include "mlp.h"

double relu(double x) 
{
    return x > 0 ? x : 0;
}

void multiply_matrices(int rowsA, int colsA, int colsB, double A[rowsA][colsA], double B[colsA][colsB], double C[rowsA][colsB]) 
{
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            C[i][j] = 0.0;
        }
    }
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

def transpose(int row, int col, double matrix[row][col], double result[row][col])
{
    
}

void forward_pass(double *input, double *w1, double *b1, double *w2, double *b2, double *output)
{

    double hidden1[HIDDEN_SIZE] = {0};

    // Couche 1
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden1[i] += input[j] * w1[i * INPUT_SIZE + j];
        }
        hidden1[i] += b1[i];
        hidden1[i] = relu(hidden1[i]);
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