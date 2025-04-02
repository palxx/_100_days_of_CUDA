
#pragma once

#include <cuda_runtime.h>

typedef struct {
 unsigned int numRows;
 unsigned int numCols;
 unsigned int numNonZeros;
 unsigned int *rowId;
 unsigned int *colId;
 float *values;
} cooMatrix;
