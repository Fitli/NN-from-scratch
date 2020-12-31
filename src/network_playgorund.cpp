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
    cout << network.get_result_xor() << endl;

    RowType i2({0, 1}); // 1
    network.load_input(i2);
    network.propagate();
    cout << network.get_result_xor() << endl;

    RowType i3({1, 0}); // 1
    network.load_input(i3);
    network.propagate();
    cout << network.get_result_xor() << endl;

    RowType i4({0, 0}); // 0
    MatrixType mt({i4});
    Matrix m(mt, 2, 1);
    network.load_input(m);
    network.propagate();
    cout << network.get_result_xor() << endl;

}

void xor_train() {

    vector<int> topology = vector<int>({2,8,1});
    NeuralNetwork network(topology, sigmoid, d_sigmoid);
    network.setLearningRate(0.1);

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
    cout << endl << "Before training evaluate on 1, 1 = 0 : " << network.get_result_xor() << endl;
    network.load_input(m_i2);
    network.propagate();
    cout << "Evaluate on 0, 1 = 1 : " << network.get_result_xor() << endl;
    network.load_input(m_i3);
    network.propagate();
    cout << "Evaluate on 1, 0 = 1 : " << network.get_result_xor() << endl;
    network.load_input(m_i4);
    network.propagate();
    cout << "Evaluate on 0, 0 = 0 : " << network.get_result_xor() << endl;

    vector<Matrix> x_train({m_i4, m_i3, m_i1, m_i2});
    vector<Matrix> y_train({ Matrix(1, 1, 0), Matrix(1, 1, 1), Matrix(1, 1, 0), Matrix(1, 1, 1)});

    srand(time(NULL));
    for(int i = 0; i < 10000; ++i) {
        int r = rand() % 4;
        network.trainOnBatch({x_train[r]}, {y_train[r]});
    }

    network.load_input(m_i1);
    network.propagate();
    cout << endl << "Evaluate on 1, 1 = 0 : " << network.get_result_xor() << endl;
    network.load_input(m_i2);
    network.propagate();
    cout << "Evaluate on 0, 1 = 1 : " << network.get_result_xor() << endl;
    network.load_input(m_i3);
    network.propagate();
    cout << "Evaluate on 1, 0 = 1 : " << network.get_result_xor() << endl;
    network.load_input(m_i4);
    network.propagate();
    cout << "Evaluate on 0, 0 = 0 : " << network.get_result_xor() << endl;

}

int main() {
    //xor_predefined_net();
    xor_train();
}
