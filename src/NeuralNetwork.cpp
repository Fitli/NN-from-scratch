//
// Created by fitli on 21.12.20.
//

#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(vector<int>& topology, float (&af) (float)):
    topology(topology),
    activation_func(af),
    num_layers(topology.size()),
    weights(vector<Matrix>(num_layers-1)),
    layers(vector<Matrix>(num_layers))
    {

    // create layers
    for(int i = 0; i < num_layers - 1; i++) {
        // last cell is bias, all cells are initialized to 1
        layers[i] = Matrix(topology[i] + 1, 1, 1);
    }
    // last layer does not have a bias.
    layers[num_layers - 1] = Matrix(topology[num_layers-1], 1, 1);

    // create weight matrices
    for(int i = 0; i < num_layers - 1; i++) {
        weights[i] = Matrix(layers[i+1].getWidth(), layers[i].getWidth());
        // Xavier Initialization of weights
        weights[i].xavier_initialization((float)(layers[i].getWidth()));
    }
    //TODO: initialize weights
}

void NeuralNetwork::propagate() {
    for(int i = 0; i < num_layers - 1; i++) {
        mul(layers[i], weights[i], layers[i+1]);
        layers[i+1].apply(activation_func);
    }
}

float NeuralNetwork::get_result() {
    return layers[num_layers - 1].get_value(0, 0);
}

void NeuralNetwork::backPropagate(RowType &result) {

}

void NeuralNetwork::learn(const string& filename_inputs, const string& filename_labels) {

}

void NeuralNetwork::label(const string& filename_input, const string& filename_ouput) {
    CSVReader input = CSVReader(filename_input);
    CSVWriter output = CSVWriter(filename_ouput);
    while(input.load_matrix(layers[0])) {
        propagate();
        float label = get_result();
        output.write_label(label);
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

void NeuralNetwork::load_input(RowType input) {
    for(int i = 0; i < input.size(); i++) {
        layers[0].put_value(input[i], 0, i);
    }
    // bias
    layers[0].put_value(1, 0, input.size());
}

void NeuralNetwork::print_weights() {
    for(int i = 0; i < num_layers - 1; ++i) {
        std::cout << "Layer " << i << " weights:" << std::endl;
        weights[i].print();
    }
}


