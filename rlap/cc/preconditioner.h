#ifndef RLAP_CC_PRECONDITIONER_H
#define RLAP_CC_PRECONDITIONER_H

#include "third_party/eigen3/Eigen/SparseCore"
#include "types.h"

class OrderedPreconditioner{
  public:
    OrderedPreconditioner(Eigen::SparseMatrix<float>* A);
    ~OrderedPreconditioner(){};
    LDLi* getLDLi();
  private:
    OrderedMatrix* getOrderedMatrix();
    float getColumnLength(OrderedMatrix* ordmat, int i, std::vector<ColumnElement>* colspace);
    void printColumn(OrderedMatrix* ordmat, int i);
    void printColspace(std::vector<ColumnElement>* colspace);
    void printMatrixDetails(OrderedMatrix* a);
    float compressColumn(std::vector<ColumnElement>* colspace, float len);
    Eigen::SparseMatrix<float>* _A;
    OrderedMatrix* _ordmat;
};

#endif
