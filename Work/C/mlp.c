#include <math.h>
#include "mlp.h"

double relu(double x) 
{
    return x > 0 ? x : 0;
}

void forward_pass(double *input, double *w1, double *b1, double *w2, double *b2, double *output)
{

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