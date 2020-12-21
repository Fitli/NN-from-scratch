//
// Created by fitli on 20.12.20.
//

#include "CSVReader.h"
#include "Matrix.h"
#include <iostream>

int main() {
    CSVReader reader = CSVReader("../../data/testing/small.csv");
    Matrix m1 = Matrix(3, 1);
    while(reader.load_matrix(m1)) {
        m1.print();
    }

    CSVReader reader2 = CSVReader("../../data/testing/small_labels.csv");
    int result;
    while(reader2.load_label(result)) {
        cout << result << endl;
    }
}
