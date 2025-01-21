#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

void load_weights(const char *filename, double *weights, int rows, int cols)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Erreur d'ouverture du fichier de poids");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows * cols; i++) {
        if (fscanf(file, "%lf", &weights[i]) != 1) {
            fprintf(stderr, "Erreur de lecture des poids\n");
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
}

void load_biases(const char *filename, double *biases, int size) 
{
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Erreur d'ouverture du fichier de biais");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%lf", &biases[i]) != 1) {
            fprintf(stderr, "Erreur de lecture des biais\n");
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
}

void load_image(const char *filename, double *image)
{

}