#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Premier main.c sur un MLP avec une seule couche cachée

// Définition des tailles
#define INPUT_SIZE 784     // 28x28 pixels pour nos images
#define HIDDEN_SIZE 128    // Taille de la couche cachée
#define OUTPUT_SIZE 10     // 10 classes pour 0 à 9

int main() 
{
    double w1[INPUT_SIZE * HIDDEN_SIZE], b1[HIDDEN_SIZE];
    double w2[HIDDEN_SIZE * HIDDEN_SIZE], b2[HIDDEN_SIZE];

    load_weights("modele/fc1_weight.txt", w1, HIDDEN_SIZE, INPUT_SIZE);
    load_biases("modele/fc1_bias.txt", b1, HIDDEN_SIZE);
    load_weights("modele/fc2_weight.txt", w2, HIDDEN_SIZE, HIDDEN_SIZE);
    load_biases("modele/fc2_bias.txt", b2, HIDDEN_SIZE);

    printf("Weights and bias loaded \n");
    printf("value of first weight : %lf \n", w1[0]);

    // Chargement de l'image
    //double image[INPUT_SIZE];
    //load_image("test_image.bmp", image);
}