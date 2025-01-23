#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utils.h"
#include "mlp.h"
#include "Bmp2Matrix.h"

// Premier main.c sur un MLP avec une seule couche cachée

// Définition des tailles
#define INPUT_SIZE 784     // 28x28 pixels pour nos images
#define HIDDEN_SIZE 128    // Taille de la couche cachée
#define OUTPUT_SIZE 10     // 10 classes pour 0 à 9

int main() 
{
    double image[INPUT_SIZE];
    double *output = (double*)malloc(OUTPUT_SIZE * sizeof(double));

    double *b1 = (double*)malloc(HIDDEN_SIZE * sizeof(double));
    double *b2 = (double*)malloc(OUTPUT_SIZE * sizeof(double));

    double **w1 = (double**)malloc(HIDDEN_SIZE*sizeof(double*));
    for(int i=0; i < INPUT_SIZE; i++)
    {
        w1[i] = (double*)malloc(INPUT_SIZE * sizeof(double));
    }
    double **w2 = (double**)malloc(OUTPUT_SIZE*sizeof(double*));
    for(int i=0; i < HIDDEN_SIZE; i++)
    {
        w2[i] = (double*)malloc(HIDDEN_SIZE * sizeof(double));
    }
    // w1 : double w1[HIDDEN_SIZE][INPUT_SIZE]
    // w2 : double w2[OUTPUT_SIZE][HIDDEN_SIZE]

    //////////////////* Chargement poids et biais *//////////////////////

    load_weights("../modele/fc1_weight.txt", w1, HIDDEN_SIZE, INPUT_SIZE);
    load_biases("../modele/fc1_bias.txt", b1, HIDDEN_SIZE);
    load_weights("../modele/fc2_weight.txt", w2, OUTPUT_SIZE, HIDDEN_SIZE);
    load_biases("../modele/fc2_bias.txt", b2, OUTPUT_SIZE);

    printf("Weights and bias loaded \n");
    printf("%lf\n", w1[0][1]);

    //////////////////* Chargement de l'image *//////////////////////
    
    BMP bitmap;
    FILE* pFichier=NULL;

    pFichier=fopen("1_1.bmp", "rb");     //Ouverture du fichier contenant l'image
    if (pFichier==NULL) {
        printf("%s\n", "0_1.bmp");
        printf("Erreur dans la lecture du fichier\n");
    }
    LireBitmap(pFichier, &bitmap);
    fclose(pFichier);               //Fermeture du fichier contenant l'image

    ConvertRGB2Gray(&bitmap);
    printf("%d\n", bitmap.mPixelsGray[9][7]);
    //DesallouerBMP(&bitmap);
    for(int j = 0; j < 28; j++)
    {
        for(int i = 0; i < 28; i++)
        {
            //printf("%d \n", j*27 + i);
            image[j*28 + i] = (double)bitmap.mPixelsGray[j][i];
        }
    }
    printf("%lf\n", image[9*28+7]);

    DesallouerBMP(&bitmap);

    //////////////////////* Phase d'inférence *//////////////////////
    printf("test\n");
    forward_pass(image, w1, b1, w2, b2, output, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

}