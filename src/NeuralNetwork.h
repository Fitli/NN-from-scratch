//
// Created by fitli on 21.12.20.
//

#ifndef SRC_NEURALNETWORK_H
#define SRC_NEURALNETWORK_H

#include <vector>

#include "Matrix.h"
#include "CSVReader.h"
#include "CSVWriter.h"


class NeuralNetwork {
    int num_layers;
    vector<int> topology;
    vector<Matrix> weights;
    vector<Matrix> layers;
    float (&activation_func) (float);
public:
    explicit NeuralNetwork(vector<int>& topology, float (&af) (float));
    void load_input(RowType input);
    void propagate();
    float get_result();
    void backPropagate(RowType &result);
    void learn(const string& filename_inputs, const string& filename_labels);
    void label(const string& filename_input, const string& filename_ouput);
    void setWeights(vector<MatrixType> weights);
    void print_weights();
};


#endif //SRC_NEURALNETWORK_H
