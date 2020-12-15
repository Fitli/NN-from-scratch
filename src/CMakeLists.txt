cmake_minimum_required(VERSION 3.17)
project(src)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(main
        main.cpp
        Matrix.cpp
        Matrix.h)
add_executable(matrix_playground
        matrix_playground.cpp
        Matrix.cpp
        Matrix.h)