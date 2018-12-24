#include "Utils.h"
#include "NetWork.h"

NetWork::NetWork(int numInputs, double learningRate)
    :mNumInputs(numInputs),
    mNumOutputs(0),
    mNumHiddenLayers(0),
    mLearningRate(learningRate),
    mErrorSum(9999) {}

NetWork::~NetWork() {
    for (size_t i = 0; i < mNeuronLayers.size(); i++) {
        if (mNeuronLayers[i]) {
            delete mNeuronLayers[i];
        }
    }
}

void NetWork::reset() {
    //for each layer
    for (int i = 0; i < mNumHiddenLayers + 1; ++i) {
        mNeuronLayers[i]->reset();
    }
    mErrorSum = 9999;
}

void NetWork::addNeuronLayer(int numNeurons) {
    int numInputsPerNeuron = (mNeuronLayers.size() > 0) ? mNeuronLayers[mNumHiddenLayers]->mNumNeurons : mNumInputs;

    /** create a neuron layer */
    mNeuronLayers.push_back(new NeuronLayer(numNeurons, numInputsPerNeuron));

    /** calculate the count of hidden layers */
    mNumHiddenLayers = (mNeuronLayers.size() > 0) ? (mNeuronLayers.size() - 1) : 0;
}

/************************************************************************
* bp neuron net forward propagation
************************************************************************/
double NetWork::sigmoidActive(double activation, double response) {
    /** sigmoid function: f(x) = 1 /(1 + exp(-x)) */
    return (1.0 / (1.0 + exp(-activation)));
}

void NetWork::updateNeuronLayer(NeuronLayer& nl, const double inputs[]) {
    int numNeurons = nl.mNumNeurons;
    int numInputsPerNeuron = nl.mNumInputsPerNeuron;
    double* curOutActivations = nl.mOutActivations;

    //for each neuron
    for (int n = 0; n < numNeurons; ++n) {
        double* curWeights = nl.mWeights[n];

        double netinput = 0;
        int k;
        //for each weight
        for (k = 0; k < numInputsPerNeuron; ++k) {
            //sum the weights x inputs
            netinput += curWeights[k] * inputs[k];
        }

        //add in the bias
        netinput += curWeights[k] * BIAS;

        //The combined activation is first filtered through the sigmoid 
        //function and a record is kept for each neuron 
        curOutActivations[n] = sigmoidActive(netinput, ACTIVATION_RESPONSE);
    }
}

void NetWork::updateNeuronLayer(NeuronLayer& nl, const int indexArray[], const size_t arraySize) {
    int numNeurons = nl.mNumNeurons;
    int numInputsPerNeuron = nl.mNumInputsPerNeuron;
    double* curOutActivations = nl.mOutActivations;

    //for each neuron
    for (int n = 0; n < numNeurons; ++n) {
        double* curWeights = nl.mWeights[n];

        double netinput = 0;
        //for each weight

        for (size_t k = 0; k < arraySize; k++) {
            netinput += curWeights[indexArray[k]];
        }

        //add in the bias
        netinput += curWeights[numInputsPerNeuron] * BIAS;


        //The combined activation is first filtered through the sigmoid 
        //function and a record is kept for each neuron 
        curOutActivations[n] = sigmoidActive(netinput, ACTIVATION_RESPONSE);
    }
}


void NetWork::process(const double inputs[], double* outputs[]) {
    for (int i = 0; i < mNumHiddenLayers + 1; i++) {
        updateNeuronLayer(*mNeuronLayers[i], inputs);
        inputs = mNeuronLayers[i]->mOutActivations;
    }

    *outputs = mNeuronLayers[mNumHiddenLayers]->mOutActivations;

}

void NetWork::process(const int indexArray[], const size_t arraySize, double* outputs[]) {

    updateNeuronLayer(*mNeuronLayers[0], indexArray, arraySize);

    double* inputs = mNeuronLayers[0]->mOutActivations;

    for (int i = 1; i < mNumHiddenLayers + 1; i++) {
        updateNeuronLayer(*mNeuronLayers[i], inputs);
        inputs = mNeuronLayers[i]->mOutActivations;
    }

    *outputs = mNeuronLayers[mNumHiddenLayers]->mOutActivations;
}

/************************************************************************
* bp neuron net back propagation
************************************************************************/

double NetWork::backActive(double x) {
    /** calculate the error value with
    * f(x) = x * (1 - x) is the derivatives of sigmoid active function
    */
    return x * (1 - x);
}

void NetWork::trainUpdate(const double inputs[], const double targets[]) {
    for (int i = 0; i < mNumHiddenLayers + 1; i++) {
        updateNeuronLayer(*mNeuronLayers[i], inputs);
        inputs = mNeuronLayers[i]->mOutActivations;
    }

    /** get the activations of output layer */
    NeuronLayer& outLayer = *mNeuronLayers[mNumHiddenLayers];
    double* outActivations = outLayer.mOutActivations;
    double* outErrors = outLayer.mOutErrors;
    int numNeurons = outLayer.mNumNeurons;
    
    mErrorSum = 0;
    /** update the out error of output neuron layer */
    for (int i = 0; i < numNeurons; i++) {
        //double err =  outActivations[i] - targets[i];
        double err = targets[i] - outActivations[i];
        outErrors[i] = err;
        /** update the SSE(Sum Squared Error). (when this value becomes lower than a
             *  preset threshold we know the training is successful)
             */
        mErrorSum += err * err;
    }
}

void NetWork::trainUpdate(const int indexArray[], const size_t arraySize, const double targets[]) {
    double* inputs;

    updateNeuronLayer(*mNeuronLayers[0], indexArray, arraySize);
    inputs = mNeuronLayers[0]->mOutActivations;


    for (int i = 1; i < mNumHiddenLayers + 1; i++) {
        updateNeuronLayer(*mNeuronLayers[i], inputs);
        inputs = mNeuronLayers[i]->mOutActivations;
    }

    /** get the activations of output layer */
    NeuronLayer& outLayer = *mNeuronLayers[mNumHiddenLayers];
    double* outActivations = outLayer.mOutActivations;
    double* outErrors = outLayer.mOutErrors;
    int numNeurons = outLayer.mNumNeurons;

    mErrorSum = 0;
    /** update the out error of output neuron layer */
    for (int i = 0; i < numNeurons; i++) {
        //double err =  outActivations[i] - targets[i];
        double err = targets[i] - outActivations[i];
        outErrors[i] = err;
        /** update the SSE(Sum Squared Error). (when this value becomes lower than a
        *  preset threshold we know the training is successful)
        */
        mErrorSum += err * err;
    }
}


void NetWork::trainNeuronLayer(NeuronLayer& nl, const double prevOutActivations[], double prevOutErrors[]) {
    int numNeurons = nl.mNumNeurons;
    int numInputsPerNeuron = nl.mNumInputsPerNeuron;
    double* curOutErrors = nl.mOutErrors;
    double* curOutActivations = nl.mOutActivations;

    /** for each neuron of current layer calculate the error and adjust weights accordingly */

    for (int i = 0; i < numNeurons; i++) {
        double* curWeights = nl.mWeights[i];
        double coi = curOutActivations[i];
        /** calculate the error value with  
         * f(x) = x * (1 - x) is the derivatives of sigmoid active function
         */
        double err = curOutErrors[i] * backActive(coi);

        /** for each weight in this neuron calculate the new weight based
         *  on the error signal and the learning rate
         */

        int w;
        //for each weight up to but not including the bias
        for (w = 0; w < numInputsPerNeuron; w++) {
            /** update the output error of prev neuron layer */
            if (prevOutErrors) /** because the input layer only have data, haven't other member */{
                prevOutErrors[w] += curWeights[w] * err;
            }	

            /** calculate the new weight based on the back propagation rules */
            curWeights[w] += err * mLearningRate * prevOutActivations[w];
        }

        /** and the bias for this neuron */
        curWeights[w] += err * mLearningRate * BIAS;
    }
}

void NetWork::trainNeuronLayer(NeuronLayer& nl, const int indexArray[], const size_t arraySize) {
    int numNeurons = nl.mNumNeurons;
    int numInputsPerNeuron = nl.mNumInputsPerNeuron;
    double* curOutErrors = nl.mOutErrors;
    double* curOutActivations = nl.mOutActivations;

    /** for each neuron of current layer calculate the error and adjust weights accordingly */

    for (int i = 0; i < numNeurons; i++) {
        double* curWeights = nl.mWeights[i];
        double coi = curOutActivations[i];
        /** calculate the error value with
        * f(x) = x * (1 - x) is the derivatives of sigmoid active function
        */
        double err = curOutErrors[i] * backActive(coi);

        /** for each weight in this neuron calculate the new weight based
        *  on the error signal and the learning rate
        */
        double deltaW = err * mLearningRate;
        //for each weight up to but not including the bias
        for (int w = 0; w < arraySize; w++)
        {
            /** calculate the new weight based on the back propagation rules */
            curWeights[indexArray[w]] += deltaW;
        }

        /** and the bias for this neuron */
        curWeights[numInputsPerNeuron] += err * mLearningRate * BIAS;
    }
}


bool NetWork::training(const double inputs[], const double targets[]) {
    const double* prevOutActivations = NULL;
    double* prevOutErrors = NULL;
    trainUpdate(inputs, targets);

    for (int i = mNumHiddenLayers; i >= 0; i--) {
        NeuronLayer& curLayer = *mNeuronLayers[i];

        /** get the out activation of prev layer or use inputs data */

        if (i > 0) {
            NeuronLayer& prev = *mNeuronLayers[(i - 1)];
            prevOutActivations = prev.mOutActivations;
            prevOutErrors = prev.mOutErrors;
            memset(prevOutErrors, 0, prev.mNumNeurons * sizeof(double));

        }
        else {
            prevOutActivations = inputs;
            prevOutErrors = NULL;
        }

        trainNeuronLayer(curLayer, prevOutActivations, prevOutErrors);
    }

    return true;
}


bool NetWork::training(const int indexArray[], const size_t arraySize, const double targets[]) {
    const double* prevOutActivations = NULL;
    double* prevOutErrors = NULL;
    trainUpdate(indexArray, arraySize, targets);

    for (int i = mNumHiddenLayers; i >= 0; i--) {
        NeuronLayer& curLayer = *mNeuronLayers[i];

        /** get the out activation of prev layer or use inputs data */

        if (i > 0) {
            NeuronLayer& prev = *mNeuronLayers[(i - 1)];
            prevOutActivations = prev.mOutActivations;
            prevOutErrors = prev.mOutErrors;
            memset(prevOutErrors, 0, prev.mNumNeurons * sizeof(double));
            trainNeuronLayer(curLayer, prevOutActivations, prevOutErrors);

        }
        else {
            trainNeuronLayer(curLayer, indexArray, arraySize);
        }        
    }

    return true;
}