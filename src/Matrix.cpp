//
// Created by fitli on 14.12.20.
//

#include "Matrix.h"

#include <utility>
#include <exception>
#include <iostream>
#include <random>
#include <cmath>
#include <chrono>


Matrix::Matrix(): Matrix(0, 0) {}

Matrix::Matrix(MatrixType matrix) : Matrix(matrix, matrix[0].size(), matrix.size()){}

Matrix::Matrix(int width, int height, float value) : Matrix(new_matrix(width, height, value), width, height){}

Matrix::Matrix(MatrixType matrix, int width, int height) :
        matrix(std::move(matrix)), transposed(nullptr), width(width), height(height){}

Matrix::Matrix(Matrix *transposed, int width, int height) : Matrix(width, height) {
    this->transposed = transposed;
    for(int row = 0; row<height; row++) {
        for (int column = 0; column < width; column++) {
            float val = transposed->get_value(column, row);
            put_value(val, row, column, false);
        }
    }
}

float Matrix::get_value(int row, int column) const {
    return matrix[row][column];
}

void Matrix::put_value(float value, int row, int column, bool to_transposed) {
    matrix[row][column] = value;
    if(to_transposed && transposed != nullptr) {
        transposed->put_value(value, column, row, false);
    }
}

void Matrix::add_value(float value, int row, int column, bool to_transposed) {
    matrix[row][column] += value;
    if(to_transposed && transposed != nullptr) {
        transposed->add_value(value, column, row, false);
    }
}

int Matrix::getHeight() const {
    return height;
}

int Matrix::getWidth() const {
    return width;
}

Matrix *Matrix::getTransposed(bool update){
    if(transposed == nullptr) {
        transposed = new Matrix(this, height, width);
        // TODO asi chce kontrolu validniho ukazatele
        // mozna by bylo lepsi make_unique<Matrix>(this, height, width), #include <memory>
        return transposed;
    }
    if(update) {
        for(int row = 0; row<getWidth(); row++) {
            for (int column = 0; column < getHeight(); column++) {
                float val = get_value(column, row);
                transposed->put_value(val, row, column, false);
            }
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

void Matrix::apply(float (&func)(float), bool to_transposed) {
    for(int row = 0; row<height; row++) {
        for(int column = 0; column<width; column++) {
            float val = this->get_value(row, column);
            this->put_value(func(val), row, column, to_transposed);
        }
    }
}

void Matrix::xavier_initialization(float n, bool to_transposed) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0.0,1.0);
    for(int row = 0; row<height; row++) {
        for(int column = 0; column<width; column++) {
            this->put_value(distribution(generator)/sqrtf(n), row, column, to_transposed);
        }
    }
}

void Matrix::set_all(float val, bool to_transposed) {
    for(int row = 0; row<height; row++) {
        for(int column = 0; column<width; column++) {
            this->put_value(val, row, column, to_transposed);
        }
    }
}

float Matrix::exp_sum() const {
    float s = 0;
    for(int i = 0; i < height; ++i) {
        for(int j = 0; j < width; ++j) {
            s += (float)(std::exp(this->get_value(i, j)));
        }
    }
    return s;
}

float Matrix::max() const {
    float max = get_value(0, 0);
    for(int i = 0; i < height; ++i) {
        for(int j = 0; j < width; ++j) {
            float val = get_value(i, j);
            if(val > max) {
                max = val;
            }
        }
    }
    return max;
}

float Matrix::min() const {
    float min = get_value(0, 0);
    for(int i = 0; i < height; ++i) {
        for(int j = 0; j < width; ++j) {
            float val = get_value(i, j);
            if(val < min) {
                min = val;
            }
        }
    }
    return min;
}

float Matrix::mean() const {
    float sum = 0;
    for(int i = 0; i < height; ++i) {
        for(int j = 0; j < width; ++j) {
            float val = get_value(i, j);
            sum += val;
        }
    }
    return sum/(getWidth() * getHeight());
}


MatrixType new_matrix(int width, int height, float val) {
    RowType row(width, val);
    MatrixType matrix(height, row);
    return matrix;
}


void sum(Matrix& first, Matrix& second, Matrix& result, bool to_transposed) {
    /*if(first.getHeight() != second.getHeight()  or first.getHeight() != result.getHeight()) {
        throw invalid_argument("Wrong matrix size.");
    }
    if(first.getWidth() != second.getWidth()  or first.getWidth() != result.getWidth()) {
        throw invalid_argument("Wrong matrix size.");
    }*/
    for(int row = 0; row < first.getHeight(); row++) {
        for(int column = 0; column < first.getWidth(); column++) {
            float val = first.get_value(row, column) + second.get_value(row, column);
            result.put_value(val, row, column, to_transposed);
        }
    }
}

void subtract(Matrix &first, Matrix &second, Matrix& result, bool to_transposed) {
    /*if(first.getHeight() != second.getHeight()  or first.getHeight() != result.getHeight()) {
        throw invalid_argument("Wrong matrix size.");
    }
    if(first.getWidth() != second.getWidth()  or first.getWidth() != result.getWidth()) {
        throw invalid_argument("Wrong matrix size.");
    }*/
    for(int row = 0; row < first.getHeight(); row++) {
        for(int column = 0; column < first.getWidth(); column++) {
            float val = first.get_value(row, column) - second.get_value(row, column);
            result.put_value(val, row, column, to_transposed);
        }
    }
}

void subtract(Matrix &first, float num, Matrix& result, bool to_transposed) {
    /*if(first.getHeight() != result.getHeight()) {
        throw invalid_argument("Wrong matrix size.");
    }
    if(first.getWidth() != result.getWidth()) {
        throw invalid_argument("Wrong matrix size.");
    }*/
    for(int row = 0; row < first.getHeight(); row++) {
        for(int column = 0; column < first.getWidth(); column++) {
            float val = first.get_value(row, column) - num;
            result.put_value(val, row, column, to_transposed);
        }
    }
}

void elem_mul(Matrix& first, Matrix& second, Matrix& result, bool to_transposed) {
    /*if(first.getHeight() != second.getHeight()  or first.getHeight() != result.getHeight()) {
        throw invalid_argument("Wrong matrix size.");
    }
    if(first.getWidth() != second.getWidth()  or first.getWidth() != result.getWidth()) {
        throw invalid_argument("Wrong matrix size.");
    }*/
    for(int row = 0; row < first.getHeight(); row++) {
        for(int column = 0; column < first.getWidth(); column++) {
            float val = first.get_value(row, column) * second.get_value(row, column);
            result.put_value(val, row, column, to_transposed);
        }
    }
}

void mul(Matrix& first, Matrix& second, Matrix& result, bool to_transposed){
    /*if(first.getWidth() != second.getHeight()) {
        throw invalid_argument("Wrong matrix size: first width x second height.");
    }
    if(first.getHeight() != result.getHeight()) {
        throw invalid_argument("Wrong matrix size: first x result.");
    }
    if(second.getWidth() != result.getWidth()) {
        throw invalid_argument("Wrong matrix size: second x result.");
    }*/

    int size = first.getWidth();
    for(int row = 0; row < first.getHeight(); row++) {
        for(int column = 0; column < second.getWidth(); column++) {
            float val = mul(first.get_row(row), second.getTransposed()->get_row(column), size);
            result.put_value(val, row, column, to_transposed);
        }
    }

}

void mul1d(Matrix& first, Matrix& second, Matrix& result, bool to_transposed, bool update_transposed) {
    RowType& col = second.get_row(0);
    RowType& row = first.getTransposed(update_transposed)->get_row(0);
    for(int r = 0; r < row.size(); r++) {
        for(int c = 0; c < col.size(); c++) {
            result.put_value(row[r]*col[c], r, c, to_transposed);
        }
    }
}

void add_mul(Matrix& first, Matrix& second, Matrix& result, bool to_transposed){
    /*if(first.getWidth() != second.getHeight()) {
        throw invalid_argument("Wrong matrix size: first width x second height.");
    }
    if(first.getHeight() != result.getHeight()) {
        throw invalid_argument("Wrong matrix size: first x result.");
    }
    if(second.getWidth() != result.getWidth()) {
        throw invalid_argument("Wrong matrix size: second x result.");
    }*/

    int size = first.getWidth();
    for(int row = 0; row < first.getHeight(); row++) {
        for(int column = 0; column < second.getWidth(); column++) {
            float val = mul(first.get_row(row), second.getTransposed()->get_row(column), size);
            result.add_value(val, row, column, to_transposed);
        }
    }

}

void add_mul1d(Matrix& first, Matrix& second, Matrix& result, bool to_transposed, bool update_transposed) {
    RowType& col = second.get_row(0);
    RowType& row = first.getTransposed(update_transposed)->get_row(0);
    for(int r = 0; r < row.size(); r++) {
        for(int c = 0; c < col.size(); c++) {
            result.add_value(row[r]*col[c], r, c, to_transposed);
        }
    }
}

void mul(Matrix &matrix, float num, Matrix &result, bool to_transposed) {
    /*if(matrix.getHeight() != result.getHeight()) {
        throw invalid_argument("Wrong matrix size.");
    }
    if(matrix.getWidth() != result.getWidth()) {
        throw invalid_argument("Wrong matrix size.");
    }*/
    for(int row = 0; row < matrix.getHeight(); row++) {
        for(int column = 0; column < matrix.getWidth(); column++) {
            float val = matrix.get_value(row, column) * num;
            result.put_value(val, row, column, to_transposed);
        }
    }
}

float mul(RowType &first, RowType &second, int size) {
    //if(first.size() != second.size()) {
    //    throw invalid_argument("Wrong matrix size.");
    //}
    float result = 0;
    for(int i = 0; i<size; i++) {
        result += first[i] * second[i];
    }
    return result;
}
