#ifndef __NEURAL__H__
#define __NEURAL__H__


class Neural{

    public:

        Neural();
        Neural(unsigned int, unsigned int, float **, float *);
        float sigmoide(float);
        float gradientSigmoide(float);
        float *training(unsigned int, float);
        void freeMemory();

    private:

        unsigned int _ligne;
        unsigned int _colonne;
        float **_input;
        float *_outputReal;
        float _biais;
        float *_poids;
        
};




#endif