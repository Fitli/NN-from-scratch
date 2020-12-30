//
// Created by fitli on 26.12.20.
//

#include "NeuralNetwork.h"
#include <cmath>

float unit_step(float in) {
    if(in >= 0) {
        return 1;
    }
    return 0;
}

float sigmoid(float in) {
    return 1 / (1 + (float)(std::exp(-in)));
}

float d_sigmoid(float in) { //hack - derivace pocita s aplikovanim sigmoidu na inner potential neuronu
    return in * (1 - in);
}

/**
 * Create network for XOR with predefined weights and test NeuralNetwork.propagate()
 */
void xor_predefined_net() {
    vector<int> topology = vector<int>({2,2,1});
    NeuralNetwork network(topology, unit_step, unit_step);

    vector<float> w11({2, -2});
    vector<float> w13({-1, 3});
    MatrixType w1({w11, w11});

    RowType w21({1});
    RowType w23({-2});
    MatrixType w2({w21, w21});

    vector<MatrixType> weights({w1, w2});
    network.setWeights(weights);

    vector<float> b1({-1, 3});
    vector<float> b2({-2});
    MatrixType rb1({b1});
    MatrixType rb2({b2});
    vector<MatrixType> b({rb1, rb2});
    network.setBias(b);


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
    MatrixType mt({i4});
    Matrix m(mt, 2, 1);
    network.load_input(m);
    network.propagate();
    cout << network.get_result() << endl;

}

void xor_train() {

    vector<int> topology = vector<int>({2,2,1});
    std::cout << std::endl << "Try to train" << std::endl;
    NeuralNetwork network(topology, sigmoid, d_sigmoid);
    //network.print_weights();

    RowType i4({0, 0}); // 0
    MatrixType mt_i4({i4});
    Matrix m_i4(mt_i4, 2, 1);

    RowType i3({1, 0}); // 1
    MatrixType mt_i3({i3});
    Matrix m_i3(mt_i3, 2, 1);

    RowType i2({0, 1}); // 1
    MatrixType mt_i2({i2});
    Matrix m_i2(mt_i2, 2, 1);

    RowType i1({1, 1}); // 0
    MatrixType mt_i1({i1});
    Matrix m_i1(mt_i1, 2, 1);

    network.load_input(m_i1);
    network.propagate();
    cout << endl << "Before training evaluate on 1, 1 = 0 : " << network.get_result() << endl;

    vector<Matrix> x_train({m_i4, m_i3, m_i2, m_i1});
    RowType y_train({0, 1, 1, 0});
    network.train_on_batch(x_train, y_train);
    //network.print_weights();


    network.load_input(m_i1);
    network.propagate();
    cout << endl << "After 1. training evaluate on 1, 1 = 0 : " << network.get_result() << endl;

    network.train_on_batch(x_train, y_train);
    network.load_input(m_i1);
    network.propagate();
    cout << endl << "After 2. training evaluate on 1, 1 = 0 : " << network.get_result() << endl;


}

int main() {
    //xor_predefined_net();
    xor_train();
}
