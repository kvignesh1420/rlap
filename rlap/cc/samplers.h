#ifndef RLAP_CC_SAMPLERS_H
#define RLAP_CC_SAMPLERS_H

#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/Eigen/SparseCholesky"
#include "third_party/eigen3/Eigen/Core"
#include <vector>

class CliqueSampler{
    // Samplers to approximate the CLIQUE induced by removing nodes in the graph
  public:
    virtual Eigen::MatrixXf sampleClique(Eigen::SparseMatrix<float>& L, int k) = 0;
};

class ExactCliqueSampler: public CliqueSampler{
  public:
    ExactCliqueSampler(){};
    ~ExactCliqueSampler(){};
    Eigen::MatrixXf sampleClique(Eigen::SparseMatrix<float>& L, int i) override;
};

class WeightedCliqueSampler: public CliqueSampler{
  public:
    WeightedCliqueSampler(){};
    ~WeightedCliqueSampler(){};
    Eigen::MatrixXf sampleClique(Eigen::SparseMatrix<float>& L, int i) override;
};


#endif