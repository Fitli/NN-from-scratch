//
// Created by fitli on 26.12.20.
//

#include "NeuralNetwork.h"
#include "activation_functions.h"
#include <tuple>
#include <chrono>

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
    network.setLearningRate(0.003);

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
    vector<tuple<Matrix, Matrix>> train({make_tuple(x_train[0], y_train[0]), make_tuple(x_train[1], y_train[1]), make_tuple(x_train[2], y_train[2]), make_tuple(x_train[3], y_train[3])});

    srand(time(NULL));
    for(int i = 0; i < 100000; ++i) {
        int r = rand() % 4;
        network.trainOnBatch(train, 0, 3);
        if(i%100 == 0) {
            //cout << "iteration: " << i;
            //network.print_errors();
            //cout << endl;
        }
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

void xor_from_file() {
    vector<int> topology = vector<int>({2,8,1});
    NeuralNetwork network(topology, sigmoid, d_sigmoid);
    network.setLearningRate(0.001);
    network.learn("../../data/xor_vectors.csv", "../../data/xor_labels.csv", 1000, 4, 1);

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

void xor_from_file2() {
    vector<int> topology = vector<int>({2,100, 100,2});
    NeuralNetwork network(topology, relu, d_relu);
    network.setLearningRate(0.001);

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
    cout << endl << "Evaluate on 1, 1 = 0 : " << network.get_label() << endl;
    cout << network.get_result().get_value(0, 0) << " " << network.get_result().get_value(0, 1)  << endl;
    network.load_input(m_i2);
    network.propagate();
    cout << "Evaluate on 0, 1 = 1 : " << network.get_label() << endl;
    cout << network.get_result().get_value(0, 0) << " " << network.get_result().get_value(0, 1)  << endl;
    network.load_input(m_i3);
    network.propagate();
    cout << "Evaluate on 1, 0 = 1 : " << network.get_label() << endl;
    cout << network.get_result().get_value(0, 0) << " " << network.get_result().get_value(0, 1)  << endl;
    network.load_input(m_i4);
    network.propagate();
    cout << "Evaluate on 0, 0 = 0 : " << network.get_label() << endl;
    cout << network.get_result().get_value(0, 0) << " " << network.get_result().get_value(0, 1)  << endl;

    cout << "Learning" << endl;
    network.learn("../../data/xor_vectors.csv", "../../data/xor_labels.csv", 2000, 4, 1);


    network.load_input(m_i1);
    network.propagate();
    cout << endl << "Evaluate on 1, 1 = 0 : " << network.get_label() << endl;
    cout << network.get_result().get_value(0, 0) << " " << network.get_result().get_value(0, 1)  << endl;
    network.load_input(m_i2);
    network.propagate();
    cout << "Evaluate on 0, 1 = 1 : " << network.get_label() << endl;
    cout << network.get_result().get_value(0, 0) << " " << network.get_result().get_value(0, 1)  << endl;
    network.load_input(m_i3);
    network.propagate();
    cout << "Evaluate on 1, 0 = 1 : " << network.get_label() << endl;
    cout << network.get_result().get_value(0, 0) << " " << network.get_result().get_value(0, 1)  << endl;
    network.load_input(m_i4);
    network.propagate();
    cout << "Evaluate on 0, 0 = 0 : " << network.get_label() << endl;
    cout << network.get_result().get_value(0, 0) << " " << network.get_result().get_value(0, 1)  << endl;
}

void fmnist_from_file() {
    auto begin = std::chrono::steady_clock::now();
    vector<int> topology = vector<int>({784, 256, 128, 10});
    NeuralNetwork network(topology, relu, d_relu);
    network.setLearningRate(0.05);
    network.learn("../../data/fashion_mnist_train_vectors.csv", "../../data/fashion_mnist_train_labels.csv", 100, 256, 0.85);
    network.label("../../data/fashion_mnist_test_vectors.csv", "../../data/actualTestPredictions.csv");
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff1 = end-begin;
    cout << "Time to train the network: " << diff1.count()/60 << " min\n";
}

int main() {
    //xor_predefined_net();
    //xor_train();
    //xor_from_file();
    //xor_from_file2();
    fmnist_from_file();
}
