#ifndef RLAP_CC_PRECONDITIONER_H
#define RLAP_CC_PRECONDITIONER_H

#include <vector>
#include "third_party/eigen3/Eigen/SparseCore"
#include "types.h"

class Preconditioner{
  public:
    virtual LDLi* getLDLi() = 0;
    virtual ~Preconditioner() = default;
};

class OrderedPreconditioner : public Preconditioner{
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

class PriorityPreconditioner : public Preconditioner{
  public:
    PriorityPreconditioner(Eigen::SparseMatrix<float>* A);
    ~PriorityPreconditioner(){};
    LDLi* getLDLi();
  private:
    PriorityMatrix* getPriorityMatrix();
    void printColumn(PriorityMatrix* pmat, int i);
    DegreePQ* getDegreePQ(std::vector<float> a);
    float DegreePQPop(DegreePQ* pq);
    void DegreePQMove(DegreePQ* pq, int i, float newkey, int oldlist, int newlist);
    void DegreePQDec(DegreePQ* pq, int i);
    void DegreePQInc(DegreePQ* pq, int i);
    std::vector<float> getFlipIndices(Eigen::SparseMatrix<float>* M);
    float getColumnLength(PriorityMatrix* pmat, int i, std::vector<PriorityElement*>* colspace);
    float compressColumn(PriorityMatrix* pmat, std::vector<PriorityElement*>* colspace, float len, DegreePQ* pq);
    void printFlipIndices(std::vector<float> fi);

    Eigen::SparseMatrix<float>* _A;
    PriorityMatrix* _pmat;
};

#endif
