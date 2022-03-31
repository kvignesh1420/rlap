#ifndef RLAP_FACTORIZERS_H
#define RLAP_FACTORIZERS_H

#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/Eigen/SparseCholesky"
#include "third_party/eigen3/Eigen/Core"
#include <vector>
#include "samplers.h"
#include "types.h"

class Factorizer{
  public:
    // Abstract base class for all factorizers
    virtual void compute() = 0;
    virtual Eigen::SparseMatrix<float> getLaplacian() = 0;
    virtual Eigen::MatrixXf getReconstructedLaplacian() = 0;
    // get the laplacian of the adjacency matrix
    Eigen::SparseMatrix<float>* computeLaplacian(Eigen::SparseMatrix<float>* Adj);
};

class EigenCholeskyLLT: public Factorizer{
  public:
    // Eigen implementation of the L * L^t cholesky factorization
    EigenCholeskyLLT(Eigen::SparseMatrix<float>* Adj);
    ~EigenCholeskyLLT(){};
    // compute the L * L^T based cholesky factorization
    void compute() override;
    // retrive the computed laplacian
    Eigen::SparseMatrix<float> getLaplacian() override;
    // retrieve the reconstructed matrix
    Eigen::MatrixXf getReconstructedLaplacian() override;
  private:
    Eigen::SparseMatrix<float>* _L;
    Eigen::LLT<Eigen::MatrixXf>* _llt;
};

class EigenCholeskyLDLT: public Factorizer{
  public:
    // Eigen implementation of the L * D * L^t cholesky factorization
    EigenCholeskyLDLT(Eigen::SparseMatrix<float>* Adj);
    ~EigenCholeskyLDLT(){};
    // compute the L * D * L^T based cholesky factorization
    void compute() override;
    // retrive the computed laplacian
    Eigen::SparseMatrix<float> getLaplacian() override;
    // retrieve the reconstructed matrix
    Eigen::MatrixXf getReconstructedLaplacian() override;
  private:
    Eigen::SparseMatrix<float>* _L;
    Eigen::LDLT<Eigen::MatrixXf>* _ldlt;
};

class ClassicCholesky: public Factorizer{
  public:
    // Standard cholesky factorization from the ground up
    ClassicCholesky(Eigen::SparseMatrix<float>* Adj);
    ~ClassicCholesky(){};
    // compute the L * L^T based cholesky factorization
    void compute() override;
    // retrive the computed laplacian
    Eigen::SparseMatrix<float> getLaplacian() override;
    // retrieve the reconstructed matrix
    Eigen::MatrixXf getReconstructedLaplacian() override;
    // retrive the factorized lower triangular matrix
    Eigen::SparseMatrix<float> getLower();
  private:
    Eigen::SparseMatrix<float>* _L;
    Eigen::SparseMatrix<float>* _G;
};

class NaiveApproximateCholesky: public Factorizer{
  public:
    // approximate cholesky factorization from the ground up
    NaiveApproximateCholesky(Eigen::SparseMatrix<float>* Adj);
    ~NaiveApproximateCholesky(){};
    // compute the approximate L * L^T based cholesky factorization.
    // Please note that this method is just for exploration and understanding
    // purposes. It doesn't scale well practically.
    void compute() override;
    // retrive the computed laplacian
    Eigen::SparseMatrix<float> getLaplacian() override;
    // retrieve the reconstructed matrix
    Eigen::MatrixXf getReconstructedLaplacian() override;
  private:
    Eigen::SparseMatrix<float>* getStar(Eigen::SparseMatrix<float>* L, int i);
    Eigen::SparseMatrix<float>* _A;
    Eigen::SparseMatrix<float>* _L;
    Eigen::SparseMatrix<float>* _G;
};

class ApproximateCholesky: public Factorizer{
  public:
    // approximate cholesky factorization from the ground up
    ApproximateCholesky(Eigen::SparseMatrix<float>* Adj);
    ~ApproximateCholesky(){};
    // compute the pre-conditioning info for solvers
    void compute() override;
    // retrive the computed laplacian
    Eigen::SparseMatrix<float> getLaplacian() override;
    // retrieve the reconstructed matrix
    Eigen::MatrixXf getReconstructedLaplacian() override;
    // retrieve the pre-conditioner
    LDLi* getPreconditioner();
  private:
    OrderedMatrix* getOrderedMatrix();
    float getColumnLength(OrderedMatrix* ordmat, int i, std::vector<ColumnElement>* colspace);
    void printColumn(OrderedMatrix* ordmat, int i);
    void printColspace(std::vector<ColumnElement>* colspace);
    void printMatrixDetails(OrderedMatrix* a);
    float compressColumn(std::vector<ColumnElement>* colspace, float len);
    LDLi* computeLDLi(OrderedMatrix* ordmat);
    Eigen::SparseMatrix<float>* _A;
    Eigen::SparseMatrix<float>* _L;
    LDLi* _ldli;
};


#endif