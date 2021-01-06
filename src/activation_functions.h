//
// Created by fitli on 03.01.21.
//

#ifndef SRC_ACTIVATION_FUNCTIONS_H
#define SRC_ACTIVATION_FUNCTIONS_H

#include "Matrix.h"

float unit_step(float in);

float sigmoid(float in);

float d_sigmoid(float in);

float relu(float in);

float d_relu(float in);

void softmax(Matrix& m);

void stable_softmax(Matrix& m);

#endif //SRC_ACTIVATION_FUNCTIONS_H
