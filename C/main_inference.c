#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>  // Pour dirname et basename
#include <math.h>

#include "Bmp2Matrix.h"

#include "cJSON.h"


typedef struct {
    char type[10];   // "Linear" pour un MLP
    int in_features;  // Taille entrée (pour Linear uniquement)
    int out_features; // Taille sortie (pour Linear uniquement)
    double **weights; // Matrice de poids
    double *bias;     // Vecteur de biais
} Layer;

typedef struct {
    int num_layers;
    Layer *layers;
} NeuralNetwork;



void load_json(const char *filename, NeuralNetwork *network) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Erreur ouverture du fichier JSON\n");
        return;
    }
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    rewind(file);

    char *data = (char *)malloc(length + 1);

    fread(data, 1, length, file);
    data[length] = '\0';
    fclose(file);

    cJSON *json = cJSON_Parse(data);
    free(data);

    // Charger l'architecture
    cJSON *arch = cJSON_GetObjectItem(json, "architecture");

    network->num_layers = cJSON_GetArraySize(arch);
    network->layers = (Layer *)malloc(network->num_layers * sizeof(Layer));

    cJSON *params = cJSON_GetObjectItem(json, "parameters");

    for (int i = 0; i < network->num_layers; i++) {
        cJSON *layer_json = cJSON_GetArrayItem(arch, i);

        strcpy(network->layers[i].type, cJSON_GetObjectItem(layer_json, "type")->valuestring);
        if (strcmp(network->layers[i].type, "Linear") == 0) {
            network->layers[i].in_features = cJSON_GetObjectItem(layer_json, "in_features")->valueint;
            network->layers[i].out_features = cJSON_GetObjectItem(layer_json, "out_features")->valueint;

            // Allocation mémoire pour les poids et biais
            network->layers[i].weights = (double **)malloc(network->layers[i].out_features * sizeof(double *));
            for (int j = 0; j < network->layers[i].out_features; j++) {
                network->layers[i].weights[j] = (double *)malloc(network->layers[i].in_features * sizeof(double));
            }
            network->layers[i].bias = (double *)malloc(network->layers[i].out_features * sizeof(double));

            // Charger les poids et biais
            char weight_key[20], bias_key[20];
            sprintf(weight_key, "fc%d.weight", i + 1);
            sprintf(bias_key, "fc%d.bias", i + 1);

            cJSON *weight_array = cJSON_GetObjectItem(params, weight_key);
            cJSON *bias_array = cJSON_GetObjectItem(params, bias_key);

            for (int j = 0; j < network->layers[i].out_features; j++) {
                cJSON *row = cJSON_GetArrayItem(weight_array, j);
                for (int k = 0; k < network->layers[i].in_features; k++) {
                    network->layers[i].weights[j][k] = cJSON_GetArrayItem(row, k)->valuedouble;
                }
            }

            for (int j = 0; j < network->layers[i].out_features; j++) {
                network->layers[i].bias[j] = cJSON_GetArrayItem(bias_array, j)->valuedouble;
            }
        }
    }

    cJSON_Delete(json);
    printf("JSON chargé avec succès !\n");
}


void relu(double *input, int size) {
    for (int i = 0; i < size; i++) {
        if (input[i] < 0) input[i] = 0;
    }
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

void mat_vec_mult(double *output, double **matrix, double *input, double *bias, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        output[i] = bias[i];
        for (int j = 0; j < cols; j++) {
            output[i] += matrix[i][j] * input[j];
        }
    }
}

void forward(NeuralNetwork *network, double *input, double *output) {
    double *current_input = input;
    double *next_input = NULL;

    for (int i = 0; i < network->num_layers; i++) {
        Layer *layer = &network->layers[i];

        if (strcmp(layer->type, "Linear") == 0) {
            next_input = (double *)malloc(layer->out_features * sizeof(double));
            mat_vec_mult(next_input, layer->weights, current_input, layer->bias, layer->out_features, layer->in_features);
            relu(next_input, layer->out_features);
        } 

        if (i > 0) free(current_input);
        current_input = next_input;
    }

    for (int i = 0; i < network->layers[network->num_layers - 1].out_features; i++) {
        output[i] = current_input[i];
    }

    free(current_input);
}


int main() {

    //////////////////* Chargement de l'architecture et des paramètres *//////////////////////
    NeuralNetwork network;
    load_json("../modele/model_data.json", &network);

    int input_size = network.layers[0].in_features;
    int output_size = network.layers[network.num_layers - 1].out_features;
    printf("input_size : %d\n", input_size);
    printf("output_size : %d\n", output_size);

    double *image = (double *)malloc(input_size * sizeof(double));
    double *output = (double *)malloc(output_size * sizeof(double));

    //////////////////* Chargement de l'image *//////////////////////
    
    const char* image_path = "../images_inference/1_6.bmp";  // Chemin de l'image

    // Copie de la chaîne de caractères car dirname et basename modifient la chaîne
    char path_copy[256];
    snprintf(path_copy, sizeof(path_copy), "%s", image_path);

    // Affiche l'image utilisée
    char* filename = basename(path_copy);
    printf("Image de test : %s\n", filename);

    BMP bitmap;
    FILE* pFichier=NULL;

    pFichier=fopen(image_path, "rb");     
    if (pFichier==NULL) {
        printf("%s\n", image_path);
        printf("Erreur dans la lecture du fichier\n");
    }
    LireBitmap(pFichier, &bitmap);
    fclose(pFichier);               

    ConvertRGB2Gray(&bitmap);
    for(int j = 0; j < 28; j++)
    {
        for(int i = 0; i < 28; i++)
        {
            image[j*28 + i] = (double)bitmap.mPixelsGray[j][i];
        }
    }

    DesallouerBMP(&bitmap);

    //////////////////////*  Phase d'inférence  *//////////////////////

    forward(&network, image, output);

    /*printf("Résultats du modèle :\n");
    for (int i = 0; i < output_size; i++) {
        printf("Classe %d : %f\n", i, output[i]);
    }*/
    printf("Classe prédite : %d\n", argmax(output, output_size));

    free(image);
    free(output);

    return 0;
}