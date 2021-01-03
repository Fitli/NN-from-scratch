//
// Created by fitli on 03.01.21.
//

#include "activation_functions.h"
#include <cmath>

float unit_step(float in) {
    if(in >= 0) {
        return 1;
    }
    return 0;
}

float sigmoid(float in) {
    return 1 / (1 + (float)(std::exp(-in)));
}

float d_sigmoid(float in) { //hack - derivace pocita s aplikovanim sigmoidu na inner potential neuronu
    return in * (1 - in);
}

float relu(float in) {
    if(in < 0) {
        return 0;
    }
    return in;
}

float d_relu(float in) { // hack - je jedno, jestli aplikujeme až po aplikaci relu nebo před ní
    if(in <= 0) {
        return 0;
    }
    return 1;
}
