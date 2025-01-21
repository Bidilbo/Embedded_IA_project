#ifndef MLP_H
#define MLP_H

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

double relu(double x);
void forward_pass(double *input, double *w1, double *b1, double *w2, double *b2, double *w3, double *b3, double *output);
int argmax(double *array, int size);

#endif