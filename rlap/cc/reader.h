#ifndef RLAP_CC_READER_H
#define RLAP_CC_READER_H

#include "third_party/eigen3/Eigen/SparseCore"
#include <string>

class Reader{
  public:
    // read data and return the Eigen matrix representation of the data
    virtual Eigen::SparseMatrix<float>* Read() = 0;
};

class TSVReader : public Reader{
  public:
    // A reader to parse tsv files and prepare the adjacency matrix
    TSVReader(std::string filename, int nrows, int ncols);
    Eigen::SparseMatrix<float>* Read() override;
  private:
    std::string _filename;
    int _nrows, _ncols;
};

#endif