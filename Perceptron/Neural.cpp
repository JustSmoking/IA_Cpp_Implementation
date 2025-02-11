#include "Neural.h"
#include <iostream>
#include <ctime>
#include <cmath>

Neural::Neural(unsigned int ligne, unsigned int colonne, float **input, float *output)
{

    std::srand(time(0));
    this->_ligne = ligne;
    this->_colonne = colonne;
    Neural::_input = new float*[_ligne];
    Neural::_outputReal = new float[_ligne];
    Neural::_poids = new float[_colonne];

    if (!Neural::_poids)
    {
        std::cerr << "Erreur lors de l'allocation du vecteur de poids " << std::endl;
        std::abort();
    }
    
    if (!Neural::_outputReal)
    {
        std::cerr << "Erreur lors de l'allocation dynamique du tableau de sortie " << std::endl;
        std::abort;
    }
    
    if (!Neural::_input)
    {
        std::cerr << "Erreur lors de l'allocation des lignes des input " << std::endl;
        std::abort();
    }
    
    for (unsigned int i = 0; i < _ligne; i++)
    {
        _outputReal[i] = output[i];

        Neural::_input[i] = new float [_colonne];
        if (!Neural::_input[i])
        {
            std::cerr << "Erreur lors de l'allocation du tableau d'input" << std::endl;
            for (unsigned j = 0; j < i; j++)
            {
                delete Neural::_input[j];
            }
            delete Neural::_input;
            Neural::_input = nullptr;
        }
        
    }

    for (unsigned int i = 0; i < this->_ligne; i++)
    {
        for (unsigned int j = 0; j < this->_colonne; j++)
        {
            Neural::_input[i][j] = input[i][j];
        }
    }
    
    for (unsigned int i = 0; i < this->_ligne; i++)
    {
        _poids[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 - 1);
    }   
    this->_biais = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 - 1);

}

float Neural::sigmoide(float x) 
{
    if (x > 20) 
    {
        return 1; 
    } else if (x < -20) 
    {
        return 0;
    }
    return 1 / (1 + exp(-x));
}

float Neural::gradientSigmoide(float x)
{
    return (Neural::sigmoide(x) * (1 - Neural::sigmoide(x)));
}

float *Neural::training(unsigned int EPOCH, float pas)
{
    for (unsigned int k = 0; k < this->_colonne; k++)
    {
        std::cout << this->_poids[k] << " ";
    }
    std::cout<< std::endl;
    float gradientDescent = 0;
    float sommePonderer = 0;
    float y_pred = 0;
    float crossEntropy = 0;
    for (unsigned int i = 0; i < EPOCH; i++)
    {
        std::cout << "========EPOCH======== " << i + 1 << std::endl;
        for (unsigned int j = 0; j < this->_ligne; j++)
        {
            for (unsigned int k = 0; k < this->_colonne; k++)
            {
                sommePonderer += (this->_input[j][k] * _poids[k]);
            }

            sommePonderer += _biais;
            y_pred = sigmoide(sommePonderer) + 1e-10;
            crossEntropy = (-_outputReal[j]) * log(y_pred) - (1 - _outputReal[j]) * log(1 - y_pred);
            gradientDescent = (y_pred - _outputReal[j]) * gradientSigmoide(y_pred);
            
            for (unsigned int k = 0; k < this->_colonne; k++)
            {
                _poids[k] = _poids[k] - (pas * gradientDescent * _input[j][k]);
            }
            _biais = _biais - (pas * gradientDescent);
            std::cout << "Entropie-croisee = " << crossEntropy << std::endl;
        }
    }
    std::cout << "Entrainement Finis" << std::endl;
    std::cout << _biais << std::endl;
    return _poids;
}

void Neural::freeMemory()
{
    for (unsigned int i = 0; i < this->_ligne; i++)
    {
        delete _input[i];
    }
    delete _input;
    _input = nullptr;

    delete[] _poids;
    _poids = nullptr;

    delete[] _outputReal;
    _outputReal = nullptr;

    if (_input == nullptr && _outputReal == nullptr && _poids == nullptr)
    {
        std::cout << "Liberation de la memoire allouee reussie !" << std::endl;
        return;
    }
}