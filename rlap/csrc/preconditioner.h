#ifndef RLAP_CC_PRECONDITIONER_H
#define RLAP_CC_PRECONDITIONER_H

#include <vector>
#include <Eigen/SparseCore>
#include "types.h"

class Preconditioner{
  public:
    virtual Eigen::MatrixXd getSchurComplement(int64_t t) = 0;
    virtual ~Preconditioner() = default;
};

class PriorityPreconditioner : public Preconditioner{
  public:
    PriorityPreconditioner(Eigen::SparseMatrix<double>* A, std::string o_n);
    ~PriorityPreconditioner(){};
    Eigen::MatrixXd getSchurComplement(int64_t t) override;

  protected:
    PriorityMatrix* getPriorityMatrix();
    void printColumn(PriorityMatrix* pmat, int64_t i);
    DegreePQ* getDegreePQ(std::vector<double> a);
    double DegreePQPop(DegreePQ* pq);
    void DegreePQMove(DegreePQ* pq, int64_t i, double newkey, int64_t oldlist, int64_t newlist);
    void DegreePQDec(DegreePQ* pq, int64_t i);
    void DegreePQInc(DegreePQ* pq, int64_t i);
    std::vector<double> getFlipIndices(Eigen::SparseMatrix<double>* M);
    double getColumnLength(PriorityMatrix* pmat, int64_t i, std::vector<PriorityElement*>* colspace);
    double compressColumn(std::vector<PriorityElement*>* colspace, double len, DegreePQ* pq);
    double compressColumnSC(std::vector<PriorityElement*>* colspace, double len, DegreePQ* pq);
    void printFlipIndices(std::vector<double> fi);

  private:
    Eigen::SparseMatrix<double>* _A;
    std::string _o_n_str;
};

class RandomPreconditioner : public Preconditioner{
  public:
    RandomPreconditioner(Eigen::SparseMatrix<double>* A, std::string o_n);
    ~RandomPreconditioner(){};
    Eigen::MatrixXd getSchurComplement(int64_t t) override;

  protected:
    PriorityMatrix* getPriorityMatrix();
    void printColumn(PriorityMatrix* pmat, int64_t i);
    RandomPQ* getRandomPQ(std::vector<double> a);
    double RandomPQPop(RandomPQ* pq);
    std::vector<double> getFlipIndices(Eigen::SparseMatrix<double>* M);
    double getColumnLength(PriorityMatrix* pmat, int64_t i, std::vector<PriorityElement*>* colspace);
    double compressColumn(std::vector<PriorityElement*>* colspace, double len);
    double compressColumnSC(std::vector<PriorityElement*>* colspace, double len);
    void printFlipIndices(std::vector<double> fi);

  private:
    Eigen::SparseMatrix<double>* _A;
    std::string _o_n_str;
};

class CoarseningPreconditioner : public PriorityPreconditioner{
  public:
    CoarseningPreconditioner(Eigen::SparseMatrix<double>* A);
    ~CoarseningPreconditioner(){};
    Eigen::MatrixXd getSchurComplement(int64_t t) override;
  private:
    Eigen::SparseMatrix<double>* _A;
};

#endif
