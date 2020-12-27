//
// Created by fitli on 14.12.20.
//

#include "Matrix.h"

#include <utility>
#include <exception>
#include <iostream>


Matrix::Matrix(): Matrix(0, 0) {}

Matrix::Matrix(MatrixType matrix) : Matrix(matrix, matrix[0].size(), matrix.size()){}

Matrix::Matrix(int width, int height, float value) : Matrix(new_matrix(width, height, value), width, height){}

Matrix::Matrix(MatrixType matrix, int width, int height) :
        matrix(std::move(matrix)), transposed(nullptr), width(width), height(height){}

Matrix::Matrix(Matrix *transposed, int width, int height) : Matrix(width, height) {
    transposed = transposed;
}

float Matrix::get_value(int row, int column) {
    return matrix[row][column];
}

void Matrix::put_value(float value, int row, int column, bool to_transposed) {
    matrix[row][column] = value;
    if(to_transposed and transposed != nullptr) {
        transposed->put_value(value, column, row, false);
    }
}

int Matrix::getHeight() const {
    return height;
}

int Matrix::getWidth() const {
    return width;
}

Matrix *Matrix::getTransposed(){
    if(transposed == nullptr) {
        transposed = new Matrix(this, height, width);
    }
    for(int row = 0; row<height; row++) {
        for(int column = 0; column<width; column++) {
            float val = get_value(row, column);
            transposed->put_value(val, column, row, false);
        }
    }
    return transposed;
}

RowType &Matrix::get_row(int row) {
    return matrix[row];
}

void Matrix::print() {
    cout << "matrix " << width << "x" << height << ":" << endl;
    for(RowType& row: matrix) {
        for(float val: row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

void Matrix::apply(float (&func)(float)) {
    for(int row = 0; row<height; row++) {
        for(int column = 0; column<width; column++) {
            float val = this->get_value(row, column);
            this->put_value(func(val), row, column);
        }
    }
}


MatrixType new_matrix(int width, int height, float val) {
    RowType row(width, 0);
    MatrixType matrix(height, row);
    return matrix;
}


void sum(Matrix& first, Matrix& second, Matrix& result) {
    if(first.getHeight() != second.getHeight()  or first.getHeight() != result.getHeight()) {
        // No idea, jak fungují výjimky v C++;
    }
    if(first.getWidth() != second.getWidth()  or first.getWidth() != result.getWidth()) {
        // No idea, jak fungují výjimky v C++;
    }
    for(int row = 0; row < first.getHeight(); row++) {
        for(int column = 0; column < first.getWidth(); column++) {
            float val = first.get_value(row, column) + second.get_value(row, column);
            result.put_value(val, row, column);
        }
    }
}

void mul(Matrix& first, Matrix& second, Matrix& result){
    if(first.getWidth() != second.getHeight()) {
        // No idea, jak fungují výjimky v C++;
    }
    if(first.getHeight() != result.getHeight()) {
        // No idea, jak fungují výjimky v C++;
    }
    if(second.getWidth() != result.getWidth()) {
        // No idea, jak fungují výjimky v C++;
    }
    for(int row = 0; row < first.getHeight(); row++) {
        for(int column = 0; column < second.getWidth(); column++) {
            float val = mul(first.get_row(row), second.getTransposed()->get_row(column));
            result.put_value(val, row, column);
        }
    }

}

void mul(Matrix &matrix, float num, Matrix &result) {
    if(matrix.getHeight() != result.getHeight()) {
        // No idea, jak fungují výjimky v C++;
    }
    if(matrix.getWidth() != result.getWidth()) {
        // No idea, jak fungují výjimky v C++;
    }
    for(int row = 0; row < matrix.getHeight(); row++) {
        for(int column = 0; column < matrix.getWidth(); column++) {
            float val = matrix.get_value(row, column) * num;
            result.put_value(val, row, column);
        }
    }
}

float mul(RowType &first, RowType &second) {
    if(first.size() != second.size()) {
        // No idea, jak fungují výjimky v C++;
    }
    float result = 0;
    for(int i = 0; i<first.size(); i++) {
        result += first[i] * second[i];
    }
    return result;
}
