//
// Created by fitli on 26.12.20.
//

#include "CSVWriter.h"

CSVWriter::CSVWriter(const string &filename): out(ofstream(filename)){}

bool CSVWriter::write_label(float &label) {
    return bool(out << label << endl);
}
