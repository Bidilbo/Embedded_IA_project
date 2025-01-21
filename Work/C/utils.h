#ifndef UTILS_H
#define UTILS_H

void load_weights(const char *filename, double *weights, int rows, int cols);
void load_biases(const char *filename, double *biases, int size);
void load_image(const char *filename, double *image);

#endif
