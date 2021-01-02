//
// Created by fitli on 26.12.20.
//
#include "CSVReader.h"
#include <fstream>
#include <sstream>
#include <iostream>

#ifndef SRC_CSVWRITER_H
#define SRC_CSVWRITER_H

class CSVWriter {
    ofstream out;
public:
    explicit CSVWriter(const string &filename);
    bool write_label(int &label);
};


#endif //SRC_CSVWRITER_H
