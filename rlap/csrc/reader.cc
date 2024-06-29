#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <vector>
#include <fstream>
#include "reader.h"

TSVReader::TSVReader(std::string filename, int64_t nrows, int64_t ncols){
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
    int64_t i, j; double v;
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

EdgeInfoMatrixReader::EdgeInfoMatrixReader(Eigen::MatrixXd edge_info, int64_t nrows, int64_t ncols){
    _edge_info = edge_info;
    _nrows = nrows;
    _ncols = ncols;
}

Eigen::SparseMatrix<double>* EdgeInfoMatrixReader::Read(){

    Eigen::SparseMatrix<double>* Adj = new Eigen::SparseMatrix<double>(_nrows, _ncols);
    std::vector<Eigen::Triplet<double> > triplets;
    int64_t edge_count = 0;
    for(int64_t p = 0; p < _edge_info.rows(); p++){
        // Do not store the 0 values as Eigen::SparseMatrix will treat it
        // as nnz even though technically the value is zero.
        if(_edge_info(p, 2) != 0){
            triplets.push_back(Eigen::Triplet<double>(
                /*row=*/_edge_info(p, 0),
                /*col=*/_edge_info(p, 1),
                /*wt=*/_edge_info(p, 2)));
            edge_count++;
        }
    }
    // std::cout << "Total number of edges in Adj = "<< edge_count;
    Adj->setFromTriplets(triplets.begin(), triplets.end());
    return Adj;
}