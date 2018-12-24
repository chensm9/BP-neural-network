#ifndef __BP_NEURON_NET_H__
#define __BP_NEURON_NET_H__

#include <vector>
#include <cstring>

#include "NeuronLayer.h"

using namespace std;

/** used in NetWork */

/** range, 0 < x <= 1.0 */
#define ACTIVATION_RESPONSE 0.7

#define BIAS                1


/** bp neuron net */
class NetWork
{
public:
    NetWork(int numInputs, double learningRate);
    ~NetWork();
public:
    inline double getError(void) { return mErrorSum; }

    bool training(const double inputs[], const double targets[]);
    void process(const double inputs[], double* outputs[]);

    void reset(void);
    void addNeuronLayer(int numNeurons);

    /** traing with the index array of valid input data */
    bool training(const int indexArray[], const size_t arraySize, const double targets[]);

    void process(const int indexArray[], const size_t arraySize, double* outputs[]);
private:


    /** Forward propagation, calculate the output of neuron net */
    inline double sigmoidActive(double activation, double response);
    

    void updateNeuronLayer(NeuronLayer& nl, const double inputs[]);

    void updateNeuronLayer(NeuronLayer& nl, const int indexArray[], const size_t arraySize);

    /** Back propagation, for training neuron net */
    inline double backActive(double x);
    void trainUpdate(const double inputs[], const double targets[]);

    void trainUpdate(const int indexArray[], const size_t arraySize, const double targets[]);


    void trainNeuronLayer(NeuronLayer& nl,  const double prevOutActivations[], double prevOutErrors[]);

    /** just for first hidden layer() */
    void trainNeuronLayer(NeuronLayer& nl, const int indexArray[], const size_t arraySize);
private:
    int mNumInputs;
    int mNumOutputs;
    int mNumHiddenLayers; /** the total layers= mNumHiddenLayers + 1; (doesn't include input layer) */
    double mLearningRate;
    double mErrorSum;
    vector<NeuronLayer*> mNeuronLayers;
};



#endif
