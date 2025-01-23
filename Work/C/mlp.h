#ifndef MLP_H
#define MLP_H

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

double relu(double x);
void multiply_matrices(int rowsA, int colsA, double **A, double *B, double * C);
//void transpose(int row, int col, double matrix[row][col], double result[row][col]);
void forward_pass(double *input, double **w1, double *b1, double **w2, double *b2, double *output, int input_size, int hidden_size, int output_size);
int argmax(double *array, int size);

#endif