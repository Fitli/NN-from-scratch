cmake_minimum_required(VERSION 3.17)
project(src)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(main
        main.cpp
        Matrix.cpp
        Matrix.h
        CSVReader.cpp
        CSVReader.h
        NeuralNetwork.cpp
        NeuralNetwork.h
        CSVWriter.cpp
        CSVWriter.h
        activation_functions.cpp activation_functions.h)
add_executable(matrix_playground
        matrix_playground.cpp
        Matrix.cpp
        Matrix.h)
add_executable(csv_playground
        CSVReader.h
        CSVReader.cpp
        Matrix.cpp
        Matrix.h
        csv_playground.cpp
        CSVWriter.cpp
        CSVWriter.h)
add_executable(network_playgorund
        CSVReader.h
        CSVReader.cpp
        Matrix.cpp
        Matrix.h
        CSVWriter.cpp
        CSVWriter.h
        NeuralNetwork.cpp
        NeuralNetwork.h
        network_playgorund.cpp
        activation_functions.cpp activation_functions.h)