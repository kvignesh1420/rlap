#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <iostream>
#include <cmath>
#include <functional>
#include <random>
#include "factorizers.h"
#include "reader.h"
#include "preconditioner.h"
#include "tracer.h"

#define PTR_RESET -1

// Factorizer

Eigen::SparseMatrix<double>* Factorizer::computeLaplacian(Eigen::SparseMatrix<double>* Adj){
    if( ! (Adj->isApprox(Adj->transpose())) ){
        std::cout << "Error: Adjacency matrix is not symmetric" ;
        exit(0);
    }
    Eigen::SparseMatrix<double>* D = new Eigen::SparseMatrix<double>(Adj->rows(), Adj->cols());
    Eigen::SparseMatrix<double>* L = new Eigen::SparseMatrix<double>(Adj->rows(), Adj->cols());
    Eigen::VectorXd v = Eigen::VectorXd::Ones(Adj->cols());
    Eigen::VectorXd res = *Adj * v;
    D->reserve(Eigen::VectorXi::Constant(Adj->cols(), 1));
    for(int64_t i = 0; i< res.size(); i++){
        D->insert(i,i) = res[i];
    }
    D->makeCompressed();
    *L = *D - *Adj;
    // check symmetry of the laplacian
    if( ! (L->isApprox(L->transpose())) ){
        std::cout << "Error: Laplacian is not symmetric" ;
        exit(0);
    }
    delete D;
    return L;
}

// ApproximateCholesky

ApproximateCholesky::ApproximateCholesky(){}

void ApproximateCholesky::setup(Eigen::MatrixXd edge_info, int64_t nrows, int64_t ncols, std::string o_v, std::string o_n){
    Reader* r = new EdgeInfoMatrixReader(edge_info, nrows, ncols);
    _A = r->Read();
    _L = this->computeLaplacian(_A);
    _o_v_str = o_v;
    _o_n_str = o_n;
    if (_prec != nullptr){
        delete _prec;
        _prec = nullptr;
    }
    if(_o_v_str == "random"){
        _prec = new RandomPreconditioner(_A, _o_n_str);
    }
    else if(_o_v_str == "degree"){
        _prec = new PriorityPreconditioner(_A, _o_n_str);
    }
    else if(_o_v_str == "coarsen"){
        _prec = new CoarseningPreconditioner(_A);
    }
}

Eigen::SparseMatrix<double> ApproximateCholesky::getAdjacencyMatrix(){
    return *_A;
}

Eigen::MatrixXd ApproximateCholesky::getSchurComplement(int64_t t){
    return _prec->getSchurComplement(t);
}

Eigen::SparseMatrix<double> ApproximateCholesky::getLaplacian(){
    return *_L;
}