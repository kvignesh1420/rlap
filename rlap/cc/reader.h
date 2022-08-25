#ifndef RLAP_CC_READER_H
#define RLAP_CC_READER_H

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/SparseCore"
#include <string>

class Reader{
  public:
    // read data and return the Eigen matrix representation of the data
    virtual Eigen::SparseMatrix<double>* Read() = 0;
};

class TSVReader : public Reader{
  public:
    // A reader to parse tsv files and prepare the adjacency matrix
    TSVReader(std::string filename, int nrows, int ncols);
    Eigen::SparseMatrix<double>* Read() override;
  private:
    std::string _filename;
    int _nrows, _ncols;
};

class EdgeInfoMatrixReader : public Reader{
  public:
    // A reader to prepare the adjacency matrix from (row, col, wt) entries
    EdgeInfoMatrixReader(Eigen::MatrixXd edge_info, int nrows, int ncols);
    Eigen::SparseMatrix<double>* Read() override;
  private:
    Eigen::MatrixXd _edge_info;
    int _nrows, _ncols;
};

#endif