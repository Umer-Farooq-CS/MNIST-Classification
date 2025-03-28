#ifndef UTILS_H
#define UTILS_H

#include "nn.h"

// Timer function
double get_time(clock_t start);

// Matrix operations
double** allocateMatrix(int rows, int cols);
void freeMatrix(double** mat, int rows);

#endif