//
// Created by fitli on 21.12.20.
//

#ifndef SRC_NEURALNETWORK_H
#define SRC_NEURALNETWORK_H


#include "Matrix.h"
#include "CSVReader.h"
#include "CSVWriter.h"


class NeuralNetwork {
    int num_layers;
    vector<int> topology;
    vector<Matrix> weights;
    vector<Matrix> deltas;
    vector<Matrix> square_gradients;
    vector<Matrix> layers;
    vector<Matrix> bias_weights;
    vector<Matrix> bias_deltas;
    vector<Matrix> bias_square_gradients;
    vector<Matrix> errors;
    float learning_rate;
    float (&activation_func) (float);
    float (&d_activation_func) (float);
public:
    explicit NeuralNetwork(vector<int>& topology, float (&af) (float), float (&daf) (float));
    void setLearningRate(float lr);
    void load_input(RowType input);
    void load_input(const Matrix& input);
    void propagate();
    Matrix get_result();
    int get_label();
    void backPropagate(Matrix& result);
    void learn(const string& filename_inputs, const string& filename_labels, int epochs, int batch_size, float lr_decrease = 1);
    void trainOnBatch(vector <tuple<Matrix, Matrix>>& input, int start, int end);
    void label(const string& filename_input, const string& filename_output);
    void setWeights(vector<MatrixType> weights);
    void setBias(vector<MatrixType> w);
    void print_weights();
    void print_errors();
    float print_validation(ostream& s, vector<tuple<Matrix, Matrix>> validation, bool human = true);
    void print_weight_stats(ostream& s);
    float get_result_xor();

    void update_weights(int batch_size);
    void update_weights_RMS(int batch_size, float beta=0.9, float ni=0.001);

    void print_layer_stats(ostream &s);
};


#endif //SRC_NEURALNETWORK_H
