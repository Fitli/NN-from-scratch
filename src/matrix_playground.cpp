//
// Created by fitli on 14.12.20.
//

#include "Matrix.h"
#include "iostream"

using namespace std;

float plus3(float val) {
    return val + 3;
}

int main() {
        RowType v1({1, 2, 3});
        RowType v2({4, 5, 6});

        MatrixType v3({v1, v2});
        Matrix m1 = Matrix(v3);
        m1.print();

    m1.getTransposed()->print();


        RowType v4({2, 2, 7});
        RowType v5({4, 3, 1});
        MatrixType v6({v4, v5});
        Matrix m2 = Matrix(v6);
        m2.print();
        Matrix m3 = Matrix(3, 2);
        m3.print();

        sum(m1, m2, m3);
        m3.print();

        mul(m1, 5, m1);
        m1.print();

        Matrix m4 = Matrix(3, 3);
        mul(*m1.getTransposed(), m3, m4);
        m4.print();

        m4.apply(plus3);
        m4.print();

    try {
        mul(m1, m1, m3);
    } catch (invalid_argument& e) {
        cout << e.what() << endl;
    }
};