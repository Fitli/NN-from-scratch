//
// Created by fitli on 21.12.20.
//

#include "NeuralNetwork.h"

#include <algorithm>


NeuralNetwork::NeuralNetwork(vector<int>& topology, float (&af) (float), float (&daf) (float)):
    topology(topology), //je potreba?
    num_layers(topology.size()),
    weights(vector<Matrix>(num_layers-1)),
    deltas(vector<Matrix>(num_layers-1)),
    layers(vector<Matrix>(num_layers)),
    bias_weights(vector<Matrix>(num_layers-1)),
    bias_deltas(vector<Matrix>(num_layers-1)),
    errors(vector<Matrix>(num_layers)),
    learning_rate(0.0),
    activation_func(af),
    d_activation_func(daf)
    {

    // create layers + errors
    for(int i = 0; i < num_layers; i++) {
        layers[i] = Matrix(topology[i], 1);
        errors[i] = Matrix(1, topology[i]);
    }

    // create weight matrices + bias error
    for(int i = 0; i < num_layers - 1; i++) {
        weights[i] = Matrix(layers[i+1].getWidth(), layers[i].getWidth());
        bias_weights[i] = Matrix(layers[i+1].getWidth(), 1);
        // Xavier Initialization of weights
        weights[i].xavier_initialization((float)(layers[i].getWidth()));
        bias_weights[i].xavier_initialization((float)(layers[i].getWidth()));

        deltas[i] = Matrix(layers[i].getWidth(), layers[i+1].getWidth());
        bias_deltas[i] = Matrix(layers[i+1].getWidth(), 1);
    }
}

void NeuralNetwork::setLearningRate(float lr) {
    learning_rate = lr;
}

void NeuralNetwork::propagate() {
    for(int i = 0; i < num_layers - 1; i++) {
        mul(layers[i], weights[i], layers[i+1]);
        sum(layers[i+1], bias_weights[i], layers[i + 1]);
        layers[i+1].apply(activation_func);
    }
}

Matrix NeuralNetwork::get_result() {
    return layers[num_layers - 1];
}

float NeuralNetwork::get_label() {
    float max_label_index = std::max_element(layers[num_layers - 1].get_row(0).begin(),layers[num_layers - 1].get_row(0).end()) - layers[num_layers - 1].get_row(0).begin();
    return max_label_index;
}

void NeuralNetwork::backPropagate(Matrix result) {
    // initialization of error
    subtract(layers[num_layers - 1], result, errors[num_layers - 1]);
    //std::cout << "Wanted result " << result.get_value(0, 0) << endl;
    //std::cout << "Guessed result " << layers[num_layers - 1].get_value(0, 0) << endl;


    for(int i = num_layers - 2; i >= 0; --i) {
        //count error
        mul(weights[i], errors[i + 1], errors[i]);

        // TODO rozmyslet kam ukladat vysledky
        //count gradient = epsilon * Error * f'(Output)
        //count delta of weights = gradient * Input
        Matrix gradient = *layers[i + 1].getTransposed();
        gradient.apply(d_activation_func);
        elem_mul(errors[i + 1], gradient, gradient);
        mul(gradient, learning_rate, gradient);

        add_mul(gradient, layers[i], deltas[i]);
        sum(bias_deltas[i], *gradient.getTransposed(), bias_deltas[i]);
    }
}

void NeuralNetwork::update_weights() {
    for(int i = 0; i < num_layers - 1; i++) {
        subtract(weights[i], *deltas[i].getTransposed(), weights[i]);
        subtract(bias_weights[i], bias_deltas[i], bias_weights[i]);
        deltas[i].set_all(0);
        bias_deltas[i].set_all(0);
    }
}

void NeuralNetwork::learn(const string& filename_inputs, const string& filename_labels) {

}

void NeuralNetwork::trainOnBatch(const vector<Matrix> &inputs, const vector<Matrix> &labels) {
    for(int i = 0; i < inputs.size(); ++i) {
        load_input(inputs[i]);
        propagate();
        backPropagate(labels[i]);
    }
    update_weights();
}

void NeuralNetwork::label(const string& filename_input, const string& filename_output) {
    CSVReader input = CSVReader(filename_input);
    CSVWriter output = CSVWriter(filename_output);
    while(input.load_matrix(layers[0])) {
        propagate();
        float label = get_label();
        output.write_label(label);
    }
}


void NeuralNetwork::load_input(const Matrix& input) {
    for(int i = 0; i < input.getWidth(); i++) {
        layers[0].put_value(input.get_value(0, i), 0, i);
    }
}


// TESTING METHODS:
void NeuralNetwork::setWeights(vector<MatrixType> weights_) {
    for(int i = 0; i < weights_.size(); i++) {
        for(int j = 0; j < weights_[i].size(); j++) {
            for(int k = 0; k < weights_[i][j].size(); k++) {
                float value = weights_[i][j][k];
                weights[i].put_value(value, j, k);
            }
        }
    }
}


void NeuralNetwork::setBias(vector<MatrixType> w) {
    for(int i = 0; i < w.size(); ++i) {
        for(int j = 0; j < w[i][0].size(); ++j) {
            float value = w[i][0][j];
            bias_weights[i].put_value(value, 0, j);
        }
    }
}


void NeuralNetwork::load_input(RowType input) {
    for(int i = 0; i < input.size(); i++) {
        layers[0].put_value(input[i], 0, i);
    }
    // bias
    layers[0].put_value(1, 0, input.size());
}

void NeuralNetwork::print_weights() {
    std::cout << endl;
    for(int i = 0; i < num_layers - 1; ++i) {
        std::cout << "Layer " << i << " weights:" << std::endl;
        weights[i].print();
    }
}

void NeuralNetwork::print_errors() {
    std::cout << endl;
    for(int i = 0; i < num_layers; ++i) {
        std::cout << "Layer " << i << " errors:" << std::endl;
        errors[i].print();
    }
}

float NeuralNetwork::get_result_xor() {
    return layers[num_layers - 1].get_value(0, 0);
}


