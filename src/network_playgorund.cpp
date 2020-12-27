//
// Created by fitli on 26.12.20.
//

#include "NeuralNetwork.h"

float unit_step(float in) {
    if(in >= 0) {
        return 1;
    }
    return 0;
}

/**
 * Create network for XOR with predefined weights and test NeuralNetwork.propagate()
 */
void xor_net() {
    vector<int> topology = vector<int>({2,2,1});
    NeuralNetwork network(topology, unit_step);
    vector<float> w11({2, -2, 0});
    vector<float> w13({-1, 3, 0});
    MatrixType w1({w11, w11, w13});

    RowType w21({1});
    RowType w23({-2});
    MatrixType w2({w21, w21, w23});

    vector<MatrixType> weights({w1, w2});
    network.setWeights(weights);

    RowType i1({1, 1}); // 0
    network.load_input(i1);
    network.propagate();
    cout << network.get_result() << endl;

    RowType i2({0, 1}); // 1
    network.load_input(i2);
    network.propagate();
    cout << network.get_result() << endl;

    RowType i3({1, 0}); // 1
    network.load_input(i3);
    network.propagate();
    cout << network.get_result() << endl;

    RowType i4({0, 0}); // 0
    network.load_input(i4);
    network.propagate();
    cout << network.get_result() << endl;
}

int main() {
    xor_net();
}
