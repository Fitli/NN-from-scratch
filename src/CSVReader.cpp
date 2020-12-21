//
// Created by fitli on 20.12.20.
//

#include "CSVReader.h"
#include <fstream>
#include <sstream>
#include <iostream>

CSVReader::CSVReader(const string &filename):in(ifstream(filename)){}

/**
 * load one input vector from a csv file and save it to `matrix`
 * @param matrix output
 * @return true if successful
 */
bool CSVReader::load_matrix(Matrix &matrix) {
    int width = matrix.getWidth();
    string line;
    getline(in, line);
    stringstream line_ss(line);
    for(int i = 0; i < width; i++) {
        string val;
        if(not getline(line_ss, val, ',')) {
            return false;
        }
        matrix.put_value(stof(val),0, i);
    }
    return true;
}

/**
 * read one label from labels csv
 * @param label output
 * @return true if successful
 */
bool CSVReader::load_label(int &label) {
    return bool(in >> label);

}
