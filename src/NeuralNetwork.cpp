//
// Created by fitli on 21.12.20.
//

#include "NeuralNetwork.h"

#include <utility>


NeuralNetwork::NeuralNetwork(vector<int>& topology, float (&af) (float), float (&daf) (float)):
    topology(topology), //je potreba?
    num_layers(topology.size()),
    weights(vector<Matrix>(num_layers-1)),
    layers(vector<Matrix>(num_layers)),
    bias_weights(vector<Matrix>(num_layers-1)),
    bias_errors(vector<Matrix>(num_layers-1)),
    errors(vector<Matrix>(num_layers)),
    activation_func(af),
    d_activation_func(daf)
    {

    // create layers + errors
    for(int i = 0; i < num_layers; i++) {
        // last cell is bias, all cells are initialized to 1
        layers[i] = Matrix(topology[i], 1, 1);
        errors[i] = Matrix(1, topology[i]);
    }

    // create weight matrices + bias error
    for(int i = 0; i < num_layers - 1; i++) {
        bias_errors[i] = Matrix(1, 1);
        weights[i] = Matrix(layers[i+1].getWidth(), layers[i].getWidth());
        bias_weights[i] = Matrix(layers[i+1].getWidth(), 1);
        // Xavier Initialization of weights
        weights[i].xavier_initialization((float)(layers[i].getWidth()));
        bias_weights[i].xavier_initialization((float)(layers[i].getWidth()));
    }
}

void NeuralNetwork::propagate() {
    for(int i = 0; i < num_layers - 1; i++) {
        mul(layers[i], weights[i], layers[i+1]);
        sum(layers[i+1], bias_weights[i], layers[i + 1]);
        layers[i+1].apply(activation_func);
    }
}

float NeuralNetwork::get_result() {
    return layers[num_layers - 1].get_value(0, 0);
    // TODO funguje jen pro XOR
}

void NeuralNetwork::backPropagate(float result) {
    // TODO inicializace chyby pro vic ouptput neuronu
    errors[num_layers - 1].put_value(get_result() - result, 0, 0);

    // TODO pridat bias
    // TODO learning rate jinde
    float learning_rate = 0.01;

    for(int i = num_layers - 2; i >= 0; --i) {
        mul(weights[i], errors[i + 1], errors[i]);
        // TODO rozmyslet kam ukladat vysledky - dalsi vektor pro gradient?
        layers[i + 1].apply(d_activation_func);

        Matrix gradient(layers[i + 1].getWidth(), errors[i + 1].getHeight());
        mul(errors[i + 1], layers[i + 1], gradient);
        mul(gradient, learning_rate, gradient);

        Matrix delta_w(layers[i].getWidth(), gradient.getHeight());
        std::cout << std::endl << " Output ";
        layers[i + 1].print();
        std::cout << " Input " <<  std::endl;
        layers[i].print();
        std::cout << " Error " << std::endl;
        errors[i + 1].print();
        std::cout << "Gradient " << std::endl;
        gradient.print();
        std::cout << "Weights " << std::endl;
        weights[i].print();
        mul(gradient, layers[i], delta_w);
        /*
        subtract(weights[i], delta_w, weights[i]); */
    }
}

void NeuralNetwork::learn(const string& filename_inputs, const string& filename_labels) {

}

void NeuralNetwork::train_on_batch(const vector<Matrix> &inputs, const RowType &labels) {
    for(int i = 0; i < inputs.size(); ++i) {
        load_input(inputs[i]);
        propagate();
        backPropagate(labels[i]);

    }
    print_errors();
}

void NeuralNetwork::label(const string& filename_input, const string& filename_output) {
    CSVReader input = CSVReader(filename_input);
    CSVWriter output = CSVWriter(filename_output);
    while(input.load_matrix(layers[0])) {
        propagate();
        float label = get_result();
        output.write_label(label);
    }
}


void NeuralNetwork::load_input(const Matrix& input) {
    for(int i = 0; i < input.getWidth(); i++) {
        layers[0].put_value(input.get_value(0, i), 0, i);
    }
    // bias
    layers[0].put_value(1, 0, input.getWidth());
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
    for(int i = 0; i < num_layers - 1; ++i) {
        std::cout << "Layer " << i << " weights:" << std::endl;
        weights[i].print();
    }
}

void NeuralNetwork::print_errors() {
    for(int i = 0; i < num_layers - 1; ++i) {
        std::cout << "Layer " << i << " errors:" << std::endl;
        errors[i].print();
    }
}


