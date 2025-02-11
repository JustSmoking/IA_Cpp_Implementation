#include "Neural.h"
#include <iostream>


int main()
{
    int l = 8;
    int c = 3;
    unsigned int LIGNE = 8;
    unsigned int COLONNE = 3;
    float **entree = new float*[l];

    for (unsigned int i = 0; i < l; i++)
    {
        entree[i] = new float[3];
    }

    float temp[8][3] = {
        {0, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 1, 0},
        {1, 1, 1}
    };

    for (unsigned int i = 0; i < LIGNE; i++) {
        for (unsigned int j = 0; j < COLONNE; j++) {
            entree[i][j] = temp[i][j];
        }
    }
    float sortieVrai[8] = {0, 1, 1, 0, 1, 0, 0, 1};

    Neural Perceptron{LIGNE, COLONNE, entree, sortieVrai};

    float *k = Perceptron.training(100, 0.001);

    for (unsigned int i = 0; i < COLONNE; i++)
    {
        std::cout << k[i] << std::endl;
    }
    Perceptron.freeMemory();
    
}