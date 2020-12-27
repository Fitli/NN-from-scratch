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
    Matrix* transposed;
    int height;
    int width;

public:
    /*
     * Create empty `Matrix` of given width and height
     */
    Matrix(int width, int height);
    /*
     * Create `Matrix` from vector of vectors of floats
     */
    explicit Matrix(MatrixType matrix);
    Matrix(MatrixType matrix, int width, int height);
    /*
     * Create transposed `Matrix` to `transposed`
     */
    Matrix(Matrix* transposed, int width, int height);

    int getHeight() const;
    int getWidth() const;

    /*
     * Return pointer to the transposed matrix. Create it in case it does not exist.
     */
    Matrix *getTransposed();

    float get_value(int row, int column) const;
    RowType & get_row(int row);

    /*
     * Put `value` to the given `row` and `column` of the matrix.
     * If `to_transposed` is true and the `Matrix` contains pointer to the transposed matrix,
     * put the value to the transposed matrix too.
     */
    void put_value(float value, int row, int column, bool to_transposed = true);

    void print();
};

MatrixType empty_matrix(int width, int height);

/*
 * Put sum of matrices `first` and `second` to matrix `result`.
 * All of the matrices have to have the same dimensions.
 */
void sum(Matrix& first, Matrix& second, Matrix& result);
/*
 * Put matrix multiplication of `first` and `second` to matrix `result`.
 * The three matrices have to have proper dimensions for matrix multiplication.
 */
void mul(Matrix& first, Matrix& second, Matrix& result);
/*
 * Put `matrix` multiplied by `num` to matrix `result`.
 * The matrices have to have the same dimensions.
 */
void mul(Matrix& matrix, float num, Matrix& result);

/*
 * return scalar multiplication of two vectors (rows)
 */
float mul(RowType &first, RowType &second);


#endif //SRC_MATRIX_H
