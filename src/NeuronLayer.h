#ifndef __NEURON_H__
#define __NEURON_H__

#include <cstring>

struct NeuronLayer  {
public:
	NeuronLayer(int numNeurons, int numInputsPerNeuron);

	NeuronLayer(NeuronLayer& nl);
	~NeuronLayer();

	void reset(void);
public:
    // 每个神经元的输入节点数量
	int mNumInputsPerNeuron;
    // 当前层神经元数量
	int mNumNeurons;

    // 权值矩阵
	double** mWeights; /** 2D array,row: indicate neuron, column: the weights per neuron */
	double* mOutActivations; /** the output activation of neuron. 1D array, the index of array indicate neuron */
	double* mOutErrors; /** the error of output of neuron. 1D array, the index of array indicate neuron */

};


#endif