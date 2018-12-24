#include "Utils.h"
#include "NeuronLayer.h"

using namespace std;

NeuronLayer::NeuronLayer(int numNeurons, int numInputsPerNeuron)
	:mNumNeurons(numNeurons),
	mNumInputsPerNeuron(numInputsPerNeuron) {
	mWeights = new double*[mNumNeurons];
	mOutActivations = new double[mNumNeurons];
	mOutErrors = new double[mNumNeurons];
	reset();
}

NeuronLayer::NeuronLayer(NeuronLayer& nl)
	:NeuronLayer(nl.mNumNeurons, nl.mNumInputsPerNeuron) {
	int copySize = mNumNeurons * sizeof(double);

	memcpy(mOutActivations, nl.mOutActivations, copySize);
	memcpy(mOutErrors, nl.mOutErrors, copySize);

	for (int i = 0; i < mNumNeurons; i++) {
		memcpy(mWeights[i], nl.mWeights[i], mNumNeurons * sizeof(double));
	}
}

// 释放内存空间
NeuronLayer::~NeuronLayer() {
	for (int i = 0; i < mNumNeurons; i++) {
		delete []mWeights[i];
	}
	delete []mWeights;
	delete []mOutActivations;
	delete []mOutErrors;
}

void NeuronLayer::reset() {
	memset(mOutActivations, 0, mNumNeurons * sizeof(double));
	memset(mOutErrors, 0, mNumNeurons * sizeof(double));

	for (int i = 0; i < mNumNeurons; ++i) {
		int numWeights = mNumInputsPerNeuron + 1;
		double* curWeights = new double[numWeights];
		mWeights[i] = curWeights;

		for (int w = 0; w < numWeights; w++) {
			double temp = RandomClamped();
			curWeights[w] = temp;
		}
	}
}
