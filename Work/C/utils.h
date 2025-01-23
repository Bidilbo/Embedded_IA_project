#ifndef UTILS_H
#define UTILS_H

void load_weights(const char *filename, double *weights, int nb_w);
void load_biases(const char *filename, double *biases, int size);
void load_image(const char *filename, double *image);

#endif
