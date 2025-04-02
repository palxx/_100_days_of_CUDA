
#include <iostream>
#include <vector>
#include "coo_matrix.h"

void coo_kernel(const cooMatrix &h_cooMat, const float *h_inVector, float *h_outVector);

int main(){
  cooMatrix mycooMat;
  mycooMat.numRows = 3;
  mycooMat.numCols = 3;
  mycooMat.numNonZeros = 4;

  std::vector<unsigned int> rowId = {0, 0, 1, 2};
  std::vector<unsigned int> colId = {0,1,1,2};
  std::vector<float> values = {10.f, 15.f, 30.f, 14.f};

  mycooMat.rowId = rowId.data();
  mycooMat.colId = colId.data();
  mycooMat.values = values.data();

  std::vector<float>invec = {1.f, 1.f, 1.f};
  std::vector<float>outVec(mycooMat.numRows, 0.f);

  coo_kernel(mycooMat, invec.data(), outVec.data());

  std::cout << "COO SpMV result:\n";
    for (size_t i = 0; i < outVec.size(); i++) {
        std::cout << "y[" << i << "] = " << outVec[i] << "\n";
    }

    return 0;
}
