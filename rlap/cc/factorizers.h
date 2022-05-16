#ifndef RLAP_CC_FACTORIZERS_H
#define RLAP_CC_FACTORIZERS_H

#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/Eigen/SparseCholesky"
#include "third_party/eigen3/Eigen/Core"
#include <vector>
#include <string>
#include "samplers.h"
#include "types.h"

class Factorizer{
  public:
    // Abstract base class for all factorizers
    virtual void compute() = 0;
    virtual Eigen::SparseMatrix<double> getLaplacian() = 0;
    virtual Eigen::MatrixXd getReconstructedLaplacian() = 0;
    // get the laplacian of the adjacency matrix
    Eigen::SparseMatrix<double>* computeLaplacian(Eigen::SparseMatrix<double>* Adj);
    virtual ~Factorizer() = default;
};

// The reference/benchmark factorizers should be implemented below

class EigenCholeskyLLT: public Factorizer{
  public:
    // Eigen implementation of the L * L^t cholesky factorization
    EigenCholeskyLLT(Eigen::SparseMatrix<double>* Adj);
    ~EigenCholeskyLLT(){};
    // compute the L * L^T based cholesky factorization
    void compute() override;
    // retrive the computed laplacian
    Eigen::SparseMatrix<double> getLaplacian() override;
    // retrieve the reconstructed matrix
    Eigen::MatrixXd getReconstructedLaplacian() override;
  private:
    Eigen::SparseMatrix<double>* _L;
    Eigen::LLT<Eigen::MatrixXd>* _llt;
};

class EigenCholeskyLDLT: public Factorizer{
  public:
    // Eigen implementation of the L * D * L^t cholesky factorization
    EigenCholeskyLDLT(Eigen::SparseMatrix<double>* Adj);
    ~EigenCholeskyLDLT(){};
    // compute the L * D * L^T based cholesky factorization
    void compute() override;
    // retrive the computed laplacian
    Eigen::SparseMatrix<double> getLaplacian() override;
    // retrieve the reconstructed matrix
    Eigen::MatrixXd getReconstructedLaplacian() override;
  private:
    Eigen::SparseMatrix<double>* _L;
    Eigen::LDLT<Eigen::MatrixXd>* _ldlt;
};

class ClassicCholesky: public Factorizer{
  public:
    // Standard cholesky factorization from the ground up
    ClassicCholesky(Eigen::SparseMatrix<double>* Adj);
    ~ClassicCholesky(){};
    // compute the L * L^T based cholesky factorization
    void compute() override;
    // retrive the computed laplacian
    Eigen::SparseMatrix<double> getLaplacian() override;
    // retrieve the reconstructed matrix
    Eigen::MatrixXd getReconstructedLaplacian() override;
    // retrive the factorized lower triangular matrix
    Eigen::SparseMatrix<double> getLower();
  private:
    Eigen::SparseMatrix<double>* _L;
    Eigen::SparseMatrix<double>* _G;
};

class NaiveApproximateCholesky: public Factorizer{
  public:
    // approximate cholesky factorization from the ground up
    NaiveApproximateCholesky(Eigen::SparseMatrix<double>* Adj);
    ~NaiveApproximateCholesky(){};
    // compute the approximate L * L^T based cholesky factorization.
    // Please note that this method is just for exploration and understanding
    // purposes. It doesn't scale well practically.
    void compute() override;
    // retrive the computed laplacian
    Eigen::SparseMatrix<double> getLaplacian() override;
    // retrieve the reconstructed matrix
    Eigen::MatrixXd getReconstructedLaplacian() override;
  private:
    Eigen::SparseMatrix<double>* getStar(Eigen::SparseMatrix<double>* L, int i);
    Eigen::SparseMatrix<double>* _A;
    Eigen::SparseMatrix<double>* _L;
    Eigen::SparseMatrix<double>* _G;
};

// end of reference/benchmark factorizers

// The main factorizers should be implemented below
// These will be exposed as python classes for interoperable
// usage with numpy/scipy.
class ApproximateCholesky: public Factorizer{
  public:
    // approximate cholesky factorization from the ground up
    ApproximateCholesky(Eigen::SparseMatrix<double>* Adj, std::string pre = "degree");
    ApproximateCholesky(Eigen::SparseMatrix<double> Adj, std::string pre = "degree");
    // A constructor which enables the user to just pass the
    // tsv/csv file for the adjacency matrix of the graph.
    // This reduces the overhead of copying scipy csc matrices
    // from the python layer to the c++ layer.
    ApproximateCholesky(std::string filename, int nrows, int ncols, std::string pre = "degree");
    ~ApproximateCholesky(){};
    // retrieve the adjancency matrix. helpful if factorizer was
    // initialized with a filename and dimension
    Eigen::SparseMatrix<double> getAdjacencyMatrix();
    // compute the pre-conditioning info for solvers
    void compute() override;
    // retrive the computed laplacian
    Eigen::SparseMatrix<double> getLaplacian() override;
    // retrieve the reconstructed matrix
    Eigen::MatrixXd getReconstructedLaplacian() override;
    // retrieve the pre-conditioner
    LDLi* getPreconditioner();
    // solve for the unknowns
    Eigen::VectorXd solve(Eigen::VectorXd b);
    // return the number of iters of the underlying pcg solver
    int getNumIters();
    // return the ratio of edges in preconditioned/original matrix.
    double getSparsityRatio();
  private:
    Eigen::SparseMatrix<double>* _A;
    Eigen::SparseMatrix<double>* _L;
    LDLi* _ldli;
    std::string _pre_str;
    int _num_iters = 0;
};


#endif