//
// Created by fitli on 14.12.20.
//


#include "NeuralNetwork.h"
#include "activation_functions.h"
#include <chrono>

using namespace std;


int main() {
    auto begin = std::chrono::steady_clock::now();
    vector<int> topology = vector<int>({784, 256, 128, 10});
    NeuralNetwork network(topology, relu, d_relu);
    network.learn("data/fashion_mnist_train_vectors.csv", "data/fashion_mnist_train_labels.csv", 100, 256);
    network.label("data/fashion_mnist_test_vectors.csv", "actualTestPredictions.csv");
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end-begin;
    cout << "Time to train the network: " << diff.count()/60 << " min\n";
}