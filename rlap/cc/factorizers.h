#ifndef RLAP_CC_FACTORIZERS_H
#define RLAP_CC_FACTORIZERS_H

#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/Eigen/SparseCholesky"
#include "third_party/eigen3/Eigen/Core"
#include <vector>
#include <string>
#include "preconditioner.h"
#include "types.h"

class Factorizer{
  public:
    // Abstract base class for all factorizers
    virtual Eigen::SparseMatrix<double> getLaplacian() = 0;
    // get the laplacian of the adjacency matrix
    Eigen::SparseMatrix<double>* computeLaplacian(Eigen::SparseMatrix<double>* Adj);
    virtual ~Factorizer() = default;
};

// The main factorizers should be implemented below
// These will be exposed as python classes for interoperable
// usage with numpy/scipy.
class ApproximateCholesky: public Factorizer{
  public:
    ApproximateCholesky();
    ~ApproximateCholesky(){
      delete _prec;
      delete _A;
      delete _L;
    };
    // Allow the user to pass a N x 3 matrix of (row, col, wt) entries of the
    // adjacency matrix and configure the state on demand via the `setup` method.
    void setup(Eigen::MatrixXd edge_info, int nrows, int ncols, std::string o_v = "random",  std::string o_n = "asc");
    // retrieve the adjancency matrix. helpful if factorizer was
    // initialized with a filename and dimension
    Eigen::SparseMatrix<double> getAdjacencyMatrix();
    // retrive the computed laplacian
    Eigen::SparseMatrix<double> getLaplacian() override;
    // retrieve the schur complements after eliminating 't' nodes
    Eigen::MatrixXd getSchurComplement(int t);
  private:
    Preconditioner* _prec = nullptr;
    Eigen::SparseMatrix<double>* _A = nullptr;
    Eigen::SparseMatrix<double>* _L = nullptr;
    std::string _o_v_str;
    std::string _o_n_str;
};


#endif