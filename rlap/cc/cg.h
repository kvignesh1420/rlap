#ifndef RLAP_CC_CG_H
#define RLAP_CC_CG_H

#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/Eigen/SparseCholesky"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/Cholesky"
#include "types.h"

class ConjugateGradient{
  public:
    ConjugateGradient(Eigen::MatrixXd M, Eigen::VectorXd b);
    ~ConjugateGradient(){};
    Eigen::VectorXd solve(double tolerance=1e-6, int max_iters=1000);
  private:
    Eigen::MatrixXd _M;
    Eigen::VectorXd _b;
};

class PConjugateGradient{
  public:
    PConjugateGradient(Eigen::SparseMatrix<double>* M, Eigen::VectorXd* b);
    ~PConjugateGradient(){};
    Eigen::VectorXd solve(double tolerance, int max_iters=1000);
    void setPreconditioner(LDLi* ldli);
    Eigen::VectorXd applyPreconditioner(Eigen::VectorXd b);
    int getNumIters();
  private:
    Eigen::SparseMatrix<double> _M;
    Eigen::VectorXd _b;
    LDLi* _ldli;
    int _num_iters = 0;
};

#endif