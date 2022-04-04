#ifndef RLAP_CC_PRECONDITIONER_H
#define RLAP_CC_PRECONDITIONER_H

#include <vector>
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

class PriorityPreconditioner{
  public:
    PriorityPreconditioner(Eigen::SparseMatrix<float>* A);
    ~PriorityPreconditioner(){};
    LDLi* getLDLi();
  private:
    LLMatp* getLLMatp();
    void printColumn(LLMatp* pmat, int i);
    ApproxCholPQ* getApproxCholPQ(std::vector<float> a);
    float approxCholPQPop(ApproxCholPQ* pq);
    void approxCholPQMove(ApproxCholPQ* pq, int i, float newkey, int oldlist, int newlist);
    void approxCholPQDec(ApproxCholPQ* pq, int i);
    void approxCholPQInc(ApproxCholPQ* pq, int i);
    std::vector<float> getFlipIndices(Eigen::SparseMatrix<float>* M);
    float getColumnLength(LLMatp* pmat, int i, std::vector<LLp*>* colspace);
    float compressColumn(LLMatp* pmat, std::vector<LLp*>* colspace, float len, ApproxCholPQ* pq);
    void printFlipIndices(std::vector<float> fi);

    Eigen::SparseMatrix<float>* _A;
    LLMatp* _pmat;
};

#endif
