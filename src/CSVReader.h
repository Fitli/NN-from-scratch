//
// Created by fitli on 20.12.20.
//

#ifndef SRC_CSVREADER_H
#define SRC_CSVREADER_H

#include "Matrix.h"
#include "fstream"

using namespace std;

class CSVReader {
    ifstream in;
public:
    explicit CSVReader(const string &filename);
    bool load_matrix(Matrix &matrix);
    bool load_label(int &label);
};


#endif //SRC_CSVREADER_H
