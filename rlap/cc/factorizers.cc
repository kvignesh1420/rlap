#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/Eigen/SparseCholesky"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/Cholesky"
#include <iostream>
#include <cmath>
#include <functional>
#include <random>
#include "factorizers.h"
#include "samplers.h"
#include "cg.h"
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
    std::cout << "Laplacian outer size:" << L->outerSize() << std::endl;
    // check symmetry of the laplacian
    Eigen::SparseMatrix<double> L_t = L->transpose();
    if((*L - L_t ).norm() != 0){
        std::cout << "Error: Laplacian is not symmetric";
        exit(0);
    }
    return L;
}

// EigenCholeskyLLT

EigenCholeskyLLT::EigenCholeskyLLT(Eigen::SparseMatrix<double>* Adj){
    _L = this->computeLaplacian(Adj);
}

void EigenCholeskyLLT::compute(){
    _llt = new Eigen::LLT<Eigen::MatrixXd>(*_L);
}

Eigen::SparseMatrix<double> EigenCholeskyLLT::getLaplacian(){
    return *_L;
}

Eigen::MatrixXd EigenCholeskyLLT::getReconstructedLaplacian(){
    return _llt->reconstructedMatrix();
}

// EigenCholeskyLDLT

EigenCholeskyLDLT::EigenCholeskyLDLT(Eigen::SparseMatrix<double>* Adj){
    _L = this->computeLaplacian(Adj);
}

void EigenCholeskyLDLT::compute(){
    _ldlt = new Eigen::LDLT<Eigen::MatrixXd>(*_L);
}

Eigen::SparseMatrix<double> EigenCholeskyLDLT::getLaplacian(){
    return *_L;
}

Eigen::MatrixXd EigenCholeskyLDLT::getReconstructedLaplacian(){
    return _ldlt->reconstructedMatrix();
}

// ClassicCholesky

ClassicCholesky::ClassicCholesky(Eigen::SparseMatrix<double>* Adj){
    _L = this->computeLaplacian(Adj);
}

void ClassicCholesky::compute(){
    _G = new Eigen::SparseMatrix<double>(_L->outerSize(), _L->outerSize());
    // _G = new Eigen::MatrixXd(_L.outerSize(), _L.outerSize());
    Eigen::SparseMatrix<double> L = *_L;
    for(int i = 0; i < L.outerSize()-1; i++){
        // NOTE: The condition is an inequality instead of a strict equality to 0 to prevent
        // numerical instability
        if(L.coeff(i,i) >= 0.00000001){
            _G->col(i) = L.col(i)/std::sqrt(L.coeff(i,i));
            L = L - L.col(i) * (L.row(i) / L.coeff(i,i));
        }
    }
}

Eigen::SparseMatrix<double> ClassicCholesky::getLaplacian(){
    return *_L;
}

Eigen::MatrixXd ClassicCholesky::getReconstructedLaplacian(){
    return (*_G) * (_G->transpose());
}

Eigen::SparseMatrix<double> ClassicCholesky::getLower(){
    return (*_G);
}

// NaiveApproximateCholesky

NaiveApproximateCholesky::NaiveApproximateCholesky(Eigen::SparseMatrix<double>* Adj){
    _A = Adj;
    _L = this->computeLaplacian(Adj);
}

void NaiveApproximateCholesky::compute(){
    _G = new Eigen::SparseMatrix<double>(_L->outerSize(), _L->outerSize());
    // _G = new Eigen::MatrixXd(_L.outerSize(), _L.outerSize());
    Eigen::SparseMatrix<double> L = *_L;
    CliqueSampler* sampler = new WeightedCliqueSampler();
    for(int i = 0; i < L.outerSize()-1; i++){
        // NOTE: The condition is an inequality instead of a strict equality to 0 to prevent
        // numerical instability
        if(L.coeff(i,i) >= 1e-8){
            _G->col(i) = L.col(i)/std::sqrt(L.coeff(i,i));
            Eigen::SparseMatrix<double> star = *getStar(&L, i);
            Eigen::MatrixXd approx_clique = sampler->sampleClique(L, i);
            // Eigen::MatrixXd exact_clique = star - L.col(i)*(L.row(i)/L.coeff(i,i));

            // std::cout << "Exact - approx clique = " << (exact_clique - approx_clique).norm() << std::endl;
            L = L - star + approx_clique;
        }
    }
}


Eigen::SparseMatrix<double>* NaiveApproximateCholesky::getStar(Eigen::SparseMatrix<double>* L, int i){

    Eigen::SparseMatrix<double> E_i(L->rows(), L->cols());
    E_i.reserve(Eigen::VectorXi::Constant(L->cols(), 2));
    for(int j = 0; j < L->cols(); j++){
        if(_L->coeff(i, j) != 0 && i!=j){
            double w = L->coeff(i, j);
            double sign = w > 0 ? 1.0 : -1.0;
            E_i.insert(i, j) = 1 * -sign * std::sqrt(std::abs(L->coeff(i, j)));
            E_i.insert(j, j) = -1 * std::sqrt(std::abs(L->coeff(i, j)));
        }
    }
    E_i.makeCompressed();
    Eigen::SparseMatrix<double>* star_i = new Eigen::SparseMatrix<double>(L->rows(), L->cols());
    *star_i = E_i * E_i.transpose();

    return star_i;
}

Eigen::SparseMatrix<double> NaiveApproximateCholesky::getLaplacian(){
    return *_L;
}

Eigen::MatrixXd NaiveApproximateCholesky::getReconstructedLaplacian(){
    return (*_G) * (_G->transpose());
}

// ApproximateCholesky

ApproximateCholesky::ApproximateCholesky(Eigen::SparseMatrix<double>* Adj, std::string pre){
    _A = Adj;
    _L = this->computeLaplacian(_A);
    _pre_str = pre;
    this->compute();
}

ApproximateCholesky::ApproximateCholesky(Eigen::SparseMatrix<double> Adj, std::string pre){
    _A = &Adj;
    _L = this->computeLaplacian(_A);
    _pre_str = pre;
    this->compute();
}

ApproximateCholesky::ApproximateCholesky(std::string filename, int nrows, int ncols, std::string pre){
    Reader* r = new TSVReader(filename, nrows, ncols);
    _A = r->Read();
    _L = this->computeLaplacian(_A);
    _pre_str = pre;
    this->compute();
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
    Preconditioner* prec;
    if(_pre_str == "order"){
        TRACER("using OrderedPreconditioner\n");
        prec = new OrderedPreconditioner(_A);
    }
    else if(_pre_str == "coarsen"){
        TRACER("using CoarseningPreconditioner\n");
        prec = new CoarseningPreconditioner(_A);
    }
    else{
        // default option
        TRACER("using PriorityPreconditioner\n");
        prec = new PriorityPreconditioner(_A);
    }
    _ldli = prec->getLDLi();
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

Eigen::SparseMatrix<double> ApproximateCholesky::getLaplacian(){
    return *_L;
}

Eigen::MatrixXd ApproximateCholesky::getReconstructedLaplacian(){
    // TODO(kvignesh1420): Placeholder method for linking
    return *_L;
}
