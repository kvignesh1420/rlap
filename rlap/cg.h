#ifndef RLAP_CG_H
#define RLAP_CG_H

#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/Eigen/SparseCholesky"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/Cholesky"
#include "types.h"

class ConjugateGradient{
  public:
    ConjugateGradient(Eigen::MatrixXf M, Eigen::VectorXf b);
    ~ConjugateGradient(){};
    Eigen::VectorXf solve(float tolerance=1e-6, int max_iters=1000);
  private:
    Eigen::MatrixXf _M;
    Eigen::VectorXf _b;
};

class PConjugateGradient{
  public:
    PConjugateGradient(Eigen::SparseMatrix<float>* M, Eigen::VectorXf* b);
    ~PConjugateGradient(){};
    Eigen::VectorXf solve(float tolerance, int max_iters=1000);
    void setPreconditioner(LDLi* ldli);
    Eigen::VectorXf applyPreconditioner(Eigen::VectorXf b);
  private:
    Eigen::SparseMatrix<float> _M;
    Eigen::VectorXf _b;
    LDLi* _ldli;
};

#endif