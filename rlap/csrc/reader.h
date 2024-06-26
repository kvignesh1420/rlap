#ifndef RLAP_CC_READER_H
#define RLAP_CC_READER_H

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <string>

class Reader{
  public:
    // read data and return the Eigen matrix representation of the data
    virtual Eigen::SparseMatrix<double>* Read() = 0;
};

class TSVReader : public Reader{
  public:
    // A reader to parse tsv files and prepare the adjacency matrix
    TSVReader(std::string filename, int64_t nrows, int64_t ncols);
    Eigen::SparseMatrix<double>* Read() override;
  private:
    std::string _filename;
    int64_t _nrows, _ncols;
};

class EdgeInfoMatrixReader : public Reader{
  public:
    // A reader to prepare the adjacency matrix from (row, col, wt) entries
    EdgeInfoMatrixReader(Eigen::MatrixXd edge_info, int64_t nrows, int64_t ncols);
    Eigen::SparseMatrix<double>* Read() override;
  private:
    Eigen::MatrixXd _edge_info;
    int64_t _nrows, _ncols;
};

#endif