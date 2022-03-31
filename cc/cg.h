#ifndef RLAP_CC_CG_H
#define RLAP_CC_CG_H

#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/Dense>
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