//
// Created by fitli on 21.12.20.
//

#include "NeuralNetwork.h"
#include "activation_functions.h"

#include <algorithm>
#include <tuple>
#include <random>

NeuralNetwork::NeuralNetwork(vector<int>& topology, float (&af) (float), float (&daf) (float)):
    topology(topology),
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

    // create weight matrices + buffers for error
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
        if(i == num_layers - 2) {
            stable_softmax(layers[i+1]);
        } else {
            layers[i+1].apply(activation_func);
        }
    }
}

Matrix NeuralNetwork::get_result() {
    return layers[num_layers - 1];
}

int NeuralNetwork::get_label() {
    int max_label_index = std::max_element(layers[num_layers - 1].get_row(0).begin(),layers[num_layers - 1].get_row(0).end()) - layers[num_layers - 1].get_row(0).begin();
    return max_label_index;
}

void NeuralNetwork::backPropagate(Matrix& result) {
    // initialization of error
    for(int i = 0; i< result.getWidth(); i++) {
        if(result.get_value(0, i) == 0) {
            float val = 1/layers[num_layers - 1].get_value(0, i);
            errors[num_layers - 1].put_value(val, i, 0);
        }
        else {
            errors[num_layers - 1].put_value(0, i, 0);
        }
    }
    //subtract(layers[num_layers - 1], result, *errors[num_layers - 1].getTransposed());
    //std::cout << "Wanted result " << result.get_value(0, 0) << endl;
    //std::cout << "Guessed result " << layers[num_layers - 1].get_value(0, 0) << endl;


    for(int i = num_layers - 2; i >= 0; --i) {
        //count error
        mul(weights[i], errors[i + 1], errors[i]);

        // TODO rozmyslet kam ukladat vysledky
        //count gradient = epsilon * Error * f'(Output)
        //count delta of weights = gradient * Input
        Matrix gradient(1, layers[i+1].getWidth());
        if(i == num_layers - 2) {
            subtract(layers[num_layers - 1], result, *gradient.getTransposed());
        } else {
            gradient = *layers[i + 1].getTransposed();
            gradient.apply(d_activation_func, false);
            elem_mul(errors[i + 1], gradient, gradient, false);
        }
        mul(gradient, learning_rate, gradient, false);

        add_mul1d(gradient, layers[i], deltas[i], false, true);
        sum(bias_deltas[i], *gradient.getTransposed(), bias_deltas[i]);
    }
}

void NeuralNetwork::update_weights(int batch_size) {
    for(int i = 0; i < num_layers - 1; i++) {
        mul(deltas[i], 1.0/batch_size, deltas[i]);
        subtract(weights[i], *deltas[i].getTransposed(true), weights[i]);
        subtract(bias_weights[i], bias_deltas[i], bias_weights[i]);
        deltas[i].set_all(0);
        bias_deltas[i].set_all(0);
    }
}

void NeuralNetwork::learn(const string& filename_inputs, const string& filename_labels, int epochs, int batch_size) {
    // read pictures and labels from files
    CSVReader pictures = CSVReader(filename_inputs);
    CSVReader labels = CSVReader(filename_labels);
    vector<tuple<Matrix, Matrix>> inputs;
    vector<tuple<Matrix, Matrix>> validation;
    int k = 0;
    int validation_size = 10000;

    while(true) {
        tuple<Matrix, Matrix> t = make_tuple(Matrix(topology[0], 1), Matrix(topology[num_layers - 1], 1, 0));

        if (!pictures.load_matrix(get<0>(t))) break;
        int l;
        if (!labels.load_label(l)) break;
        get<1>(t).put_value(1, 0, l); //for MNIST TODO
        //get<1>(t).put_value(l, 0, 0); //for XOR

        if(k >= 60000 - validation_size) {
            validation.push_back(move(t));
        } else {
            inputs.push_back(move(t));
        }
        k++;
    }

    // train
    std::random_device rd;
    std::mt19937 g(rd());

    //ofstream f("../../data/accuracies2.csv");

    for(int i = 0; i < epochs; ++i) {


         cout << "Starting epoch " << i + 1 << " out of " << epochs << endl;
        std::shuffle(std::begin(inputs), std::end(inputs), g);
        for(int b = 0; (b + batch_size) <= inputs.size(); b+=batch_size) {
            //cout << "Starting batch " << (b/batch_size) + 1 << " out of " << inputs.size()/batch_size << endl;
            trainOnBatch(inputs, b, b + batch_size);
            //print_validation(f, validation, false);
        }

        print_validation(std::cout, validation, true);

    }

}

void NeuralNetwork::trainOnBatch(vector <tuple<Matrix, Matrix>>& input, int start, int end) {
    for(int i = start; i < end; ++i) {
        load_input(get<0>(input[i]));
        propagate();
        backPropagate(get<1>(input[i]));
    }
    update_weights(input.size());
}

void NeuralNetwork::label(const string& filename_input, const string& filename_output) {
    CSVReader input = CSVReader(filename_input);
    CSVWriter output = CSVWriter(filename_output);
    while(input.load_matrix(layers[0])) {
        propagate();
        int label = get_label();
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

void NeuralNetwork::print_validation(ostream& s, vector<tuple<Matrix, Matrix>> validation, bool human) {
    // validate
    int correct = 0;
    for(auto& value: validation) {
        load_input(get<0>(value));
        propagate();
        if(get_label() == std::max_element(get<1>(value).get_row(0).begin(),get<1>(value).get_row(0).end()) - get<1>(value).get_row(0).begin()) {
            ++correct;
        }
    }
    if(human) {
        s << "Accuracy on validation set of size " << validation.size() << " : " << (correct * 1.0) / validation.size() << endl;
    }
    else {
        s << (correct * 1.0) / validation.size() << endl;
    }
}

void NeuralNetwork::print_weight_stats(ostream& s) {
    for(int i = 0; i < num_layers - 1; i++) {
        s << "weights between layers " << i << "-" << i+1 << endl;
        s << "\tMin:" << weights[i].min();
        s << "\tMax:" << weights[i].max();
        s << "\tMean:" << weights[i].mean();
    }
}


