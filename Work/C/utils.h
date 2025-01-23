#ifndef UTILS_H
#define UTILS_H

#define INPUT_SIZE 784     // 28x28 pixels pour nos images
#define HIDDEN_SIZE 128    // Taille de la couche cachée
#define OUTPUT_SIZE 10     // 10 classes pour 0 à 9

void load_weights(const char *filename, double ** weights, int rows, int cols);
void load_biases(const char *filename, double *biases, int size);

#endif
