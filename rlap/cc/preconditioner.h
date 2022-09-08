#ifndef RLAP_CC_PRECONDITIONER_H
#define RLAP_CC_PRECONDITIONER_H

#include <vector>
#include "third_party/eigen3/Eigen/SparseCore"
#include "types.h"

class Preconditioner{
  public:
    virtual LDLi* getLDLi() = 0;
    virtual Eigen::MatrixXd getSchurComplement(int t) = 0;
    virtual ~Preconditioner() = default;
};

class OrderedPreconditioner : public Preconditioner{
  public:
    OrderedPreconditioner(Eigen::SparseMatrix<double>* A);
    ~OrderedPreconditioner(){};
    LDLi* getLDLi() override;
    Eigen::MatrixXd getSchurComplement(int t) override;
  private:
    OrderedMatrix* getOrderedMatrix();
    double getColumnLength(OrderedMatrix* ordmat, int i, std::vector<ColumnElement>* colspace);
    void printColumn(OrderedMatrix* ordmat, int i);
    void printColspace(std::vector<ColumnElement>* colspace);
    void printMatrixDetails(OrderedMatrix* a);
    double compressColumn(std::vector<ColumnElement>* colspace, double len);
    Eigen::SparseMatrix<double>* _A;
    OrderedMatrix* _ordmat;
};

class PriorityPreconditioner : public Preconditioner{
  public:
    PriorityPreconditioner(Eigen::SparseMatrix<double>* A);
    ~PriorityPreconditioner(){};
    LDLi* getLDLi() override;
    Eigen::MatrixXd getSchurComplement(int t) override;

  protected:
    PriorityMatrix* getPriorityMatrix();
    void printColumn(PriorityMatrix* pmat, int i);
    DegreePQ* getDegreePQ(std::vector<double> a);
    double DegreePQPop(DegreePQ* pq);
    void DegreePQMove(DegreePQ* pq, int i, double newkey, int oldlist, int newlist);
    void DegreePQDec(DegreePQ* pq, int i);
    void DegreePQInc(DegreePQ* pq, int i);
    std::vector<double> getFlipIndices(Eigen::SparseMatrix<double>* M);
    double getColumnLength(PriorityMatrix* pmat, int i, std::vector<PriorityElement*>* colspace);
    double compressColumn(PriorityMatrix* pmat, std::vector<PriorityElement*>* colspace, double len, DegreePQ* pq);
    void printFlipIndices(std::vector<double> fi);

  private:
    Eigen::SparseMatrix<double>* _A;
};

class RandomPreconditioner : public Preconditioner{
  public:
    RandomPreconditioner(Eigen::SparseMatrix<double>* A);
    ~RandomPreconditioner(){};
    LDLi* getLDLi() override;
    Eigen::MatrixXd getSchurComplement(int t) override;

  protected:
    PriorityMatrix* getPriorityMatrix();
    void printColumn(PriorityMatrix* pmat, int i);
    RandomPQ* getRandomPQ(std::vector<double> a);
    double RandomPQPop(RandomPQ* pq);
    std::vector<double> getFlipIndices(Eigen::SparseMatrix<double>* M);
    double getColumnLength(PriorityMatrix* pmat, int i, std::vector<PriorityElement*>* colspace);
    double compressColumn(PriorityMatrix* pmat, std::vector<PriorityElement*>* colspace, double len);
    void printFlipIndices(std::vector<double> fi);

  private:
    Eigen::SparseMatrix<double>* _A;
};

class CoarseningPreconditioner : public PriorityPreconditioner{
  public:
    CoarseningPreconditioner(Eigen::SparseMatrix<double>* A);
    ~CoarseningPreconditioner(){};
    LDLi* getLDLi() override;
    Eigen::MatrixXd getSchurComplement(int t) override;
  private:
    Eigen::SparseMatrix<double>* _A;
};

#endif
