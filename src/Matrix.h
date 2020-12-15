//
// Created by fitli on 14.12.20.
//
#include <vector>
#include <ostream>

#ifndef SRC_MATRIX_H
#define SRC_MATRIX_H

using namespace std;

typedef vector<float> RowType;
typedef vector<RowType> MatrixType;

class Matrix {
    MatrixType matrix;
    Matrix* transponed;
    int height;
    int width;

public:
    Matrix(int width, int height);
    explicit Matrix(MatrixType matrix);
    Matrix(MatrixType matrix, int width, int height);
    Matrix(Matrix* transponed, int width, int height);

    int getHeight() const;
    int getWidth() const;
    Matrix *getTransponed();

    float get_value(int row, int column);
    RowType &get_row(int row);
    void put_value(float value, int row, int column, bool to_transponed = true);

    void print();
};

MatrixType empty_matrix(int width, int height);
void sum(Matrix& first, Matrix& second, Matrix& result);
void mul(Matrix& first, Matrix& second, Matrix& result);
void mul(Matrix& matrix, float num, Matrix& result);

float mul(RowType &first, RowType &second);


#endif //SRC_MATRIX_H
