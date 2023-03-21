#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/Eigen/SparseCholesky"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/Cholesky"
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
    Eigen::SparseMatrix<double>* D = new Eigen::SparseMatrix<double>(Adj->rows(), Adj->cols());
    Eigen::SparseMatrix<double>* L = new Eigen::SparseMatrix<double>(Adj->rows(), Adj->cols());
    Eigen::VectorXd v = Eigen::VectorXd::Ones(Adj->cols());
    Eigen::VectorXd res = *Adj * v;
    D->reserve(Eigen::VectorXi::Constant(Adj->cols(), 1));
    for(int i = 0; i< res.size(); i++){
        D->insert(i,i) = res[i];
    }
    D->makeCompressed();
    *L = *D - *Adj;
    // check symmetry of the laplacian
    Eigen::SparseMatrix<double> L_t = L->transpose();
    if((*L - L_t ).norm() != 0){
        std::cout << "Error: Laplacian is not symmetric";
        exit(0);
    }
    delete D;
    return L;
}

// ApproximateCholesky

ApproximateCholesky::ApproximateCholesky(){}

void ApproximateCholesky::setup(Eigen::MatrixXd edge_info, int nrows, int ncols, std::string o_v, std::string o_n){
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

Eigen::VectorXd ApproximateCholesky::solve(Eigen::VectorXd b){
    PConjugateGradient pcg = PConjugateGradient(_L, &b);
    pcg.setPreconditioner(_ldli);
    Eigen::VectorXd x = pcg.solve(1e-12, 5000);
    _num_iters = pcg.getNumIters();
    return x;
}

void ApproximateCholesky::compute(){
    if(_prec != nullptr){
        delete _prec;
        _prec = nullptr;
    }
    if(_o_v_str == "coarsen"){
        TRACER("using CoarseningPreconditioner\n");
        _prec = new CoarseningPreconditioner(_A);
    }
    else{
        // default option
        TRACER("using PriorityPreconditioner\n");
        _prec = new PriorityPreconditioner(_A);
    }
    _ldli = _prec->getLDLi();
    std::cout << "ratio of preconditioned egdes to original edges = " << 2*float(_ldli->fval.size())/_A->nonZeros() << std::endl;
}

int ApproximateCholesky::getNumIters(){
    return _num_iters;
}

double ApproximateCholesky::getSparsityRatio(){
    return 2*float(_ldli->fval.size())/_A->nonZeros();
}

LDLi* ApproximateCholesky::getPreconditioner(){
    return _ldli;
}

Eigen::MatrixXd ApproximateCholesky::getSchurComplement(int t){
    return _prec->getSchurComplement(t);
}

Eigen::SparseMatrix<double> ApproximateCholesky::getLaplacian(){
    return *_L;
}

Eigen::MatrixXd ApproximateCholesky::getReconstructedLaplacian(){
    return *_L;
}

Eigen::MatrixXd ApproximateCholesky::getSchurComplement(int t){
    return _prec->getSchurComplement(t);
}