//
// Created by fitli on 14.12.20.
//

#include "Matrix.h"

using namespace std;

int main() {
        RowType v1({1, 2, 3});
        RowType v2({4, 5, 6});

        MatrixType v3({v1, v2});
        Matrix m1 = Matrix(v3);
        m1.print();

        m1.getTransponed()->print();


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
        mul(*m1.getTransponed(),m3, m4);
        m4.print();
};