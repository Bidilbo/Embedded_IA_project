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
    int nb_w1 = INPUT_SIZE * HIDDEN_SIZE, nb_w2 = HIDDEN_SIZE * OUTPUT_SIZE; 
    double w1[nb_w1], b1[HIDDEN_SIZE];
    double w2[nb_w2], b2[OUTPUT_SIZE];
    int image[INPUT_SIZE];

    // Pour la suite ce que je te donne en entrée gros chien (prends pas en compte avant)
    //double w1[HIDDEN_SIZE][INPUT_SIZE], b1[HIDDEN_SIZE];
    //double w2[OUTPUT_SIZE][HIDDEN_SIZE], b2[OUTPUT_SIZE];

    //////////////////* Chargement poids et biais *//////////////////////

    load_weights("../modele/fc1_weight.txt", w1, nb_w1);
    load_biases("../modele/fc1_bias.txt", b1, HIDDEN_SIZE);
    load_weights("../modele/fc2_weight.txt", w2, nb_w2);
    load_biases("../modele/fc2_bias.txt", b2, OUTPUT_SIZE);

    printf("Weights and bias loaded \n");

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
            image[j*28 + i] = bitmap.mPixelsGray[j][i];
        }
    }
    printf("%d\n", image[9*28+7]);

    DesallouerBMP(&bitmap);
    return 0;

}