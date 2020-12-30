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
    vector<Matrix> bias_weights;
    vector<Matrix> errors;
    float (&activation_func) (float);
    float (&d_activation_func) (float);
public:
    explicit NeuralNetwork(vector<int>& topology, float (&af) (float), float (&daf) (float));
    void load_input(RowType input);
    void load_input(const Matrix& input);
    void propagate();
    float get_result();
    void backPropagate(float result);
    void learn(const string& filename_inputs, const string& filename_labels);
    void train_on_batch(const vector<Matrix> &inputs, const RowType& labels);
    void label(const string& filename_input, const string& filename_output);
    void setWeights(vector<MatrixType> weights);
    void setBias(vector<MatrixType> w);
    void print_weights();
    void print_errors();
};


#endif //SRC_NEURALNETWORK_H
