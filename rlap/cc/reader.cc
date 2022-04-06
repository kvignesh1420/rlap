#include "third_party/eigen3/Eigen/SparseCore"
#include <iostream>
#include <vector>
#include <fstream>
#include "reader.h"

TSVReader::TSVReader(std::string filename, int nrows, int ncols){
    _filename = filename;
    _nrows = nrows;
    _ncols = ncols;
}

Eigen::SparseMatrix<double>* TSVReader::Read(){
    std::ifstream _ifile;

    _ifile.open(_filename);
    if (!_ifile.is_open()){
        std::cerr << "Error: The input file couldn't be opened" << std::endl;
        exit(1);
    }
    Eigen::SparseMatrix<double>* Adj = new Eigen::SparseMatrix<double>(_nrows, _ncols);
    std::vector<Eigen::Triplet<double> > triplets;
    int i, j; double v;
    while(_ifile >> i >> j >> v){
        // Do not store the 0 values as Eigen::SparseMatrix will treat it
        // as nnz even though technically the value is zero.
        if(v != 0){
            triplets.push_back(Eigen::Triplet<double>(i-1, j-1, v));
        }
    }
    Adj->setFromTriplets(triplets.begin(), triplets.end());
    return Adj;
}