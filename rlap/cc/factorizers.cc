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

#define PTR_RESET -1

// Factorizer

Eigen::SparseMatrix<float>* Factorizer::computeLaplacian(Eigen::SparseMatrix<float>* Adj){
    Eigen::SparseMatrix<float>* D = new Eigen::SparseMatrix<float>(Adj->rows(), Adj->cols());
    Eigen::SparseMatrix<float>* L = new Eigen::SparseMatrix<float>(Adj->rows(), Adj->cols());
    Eigen::VectorXf v = Eigen::VectorXf::Ones(Adj->cols());
    Eigen::VectorXf res = *Adj * v;
    D->reserve(Eigen::VectorXi::Constant(Adj->cols(), 1));
    for(int i = 0; i< res.size(); i++){
        D->insert(i,i) = res[i];
    }
    D->makeCompressed();
    *L = *D - *Adj;
    std::cout << "Laplacian outer size:" << L->outerSize() << std::endl;
    // check symmetry of the laplacian
    Eigen::SparseMatrix<float> L_t = L->transpose();
    if((*L - L_t ).norm() != 0){
        std::cout << "Error: Laplacian is not symmetric";
        exit(0);
    }
    return L;
}

// EigenCholeskyLLT

EigenCholeskyLLT::EigenCholeskyLLT(Eigen::SparseMatrix<float>* Adj){
    _L = this->computeLaplacian(Adj);
}

void EigenCholeskyLLT::compute(){
    _llt = new Eigen::LLT<Eigen::MatrixXf>(*_L);
}

Eigen::SparseMatrix<float> EigenCholeskyLLT::getLaplacian(){
    return *_L;
}

Eigen::MatrixXf EigenCholeskyLLT::getReconstructedLaplacian(){
    return _llt->reconstructedMatrix();
}

// EigenCholeskyLDLT

EigenCholeskyLDLT::EigenCholeskyLDLT(Eigen::SparseMatrix<float>* Adj){
    _L = this->computeLaplacian(Adj);
}

void EigenCholeskyLDLT::compute(){
    _ldlt = new Eigen::LDLT<Eigen::MatrixXf>(*_L);
}

Eigen::SparseMatrix<float> EigenCholeskyLDLT::getLaplacian(){
    return *_L;
}

Eigen::MatrixXf EigenCholeskyLDLT::getReconstructedLaplacian(){
    return _ldlt->reconstructedMatrix();
}

// ClassicCholesky

ClassicCholesky::ClassicCholesky(Eigen::SparseMatrix<float>* Adj){
    _L = this->computeLaplacian(Adj);
}

void ClassicCholesky::compute(){
    _G = new Eigen::SparseMatrix<float>(_L->outerSize(), _L->outerSize());
    // _G = new Eigen::MatrixXf(_L.outerSize(), _L.outerSize());
    Eigen::SparseMatrix<float> L = *_L;
    for(int i = 0; i < L.outerSize()-1; i++){
        // NOTE: The condition is an inequality instead of a strict equality to 0 to prevent
        // numerical instability
        if(L.coeff(i,i) >= 0.00000001){
            _G->col(i) = L.col(i)/std::sqrt(L.coeff(i,i));
            L = L - L.col(i) * (L.row(i) / L.coeff(i,i));
        }
    }
}

Eigen::SparseMatrix<float> ClassicCholesky::getLaplacian(){
    return *_L;
}

Eigen::MatrixXf ClassicCholesky::getReconstructedLaplacian(){
    return (*_G) * (_G->transpose());
}

Eigen::SparseMatrix<float> ClassicCholesky::getLower(){
    return (*_G);
}

// NaiveApproximateCholesky

NaiveApproximateCholesky::NaiveApproximateCholesky(Eigen::SparseMatrix<float>* Adj){
    _A = Adj;
    _L = this->computeLaplacian(Adj);
}

void NaiveApproximateCholesky::compute(){
    _G = new Eigen::SparseMatrix<float>(_L->outerSize(), _L->outerSize());
    // _G = new Eigen::MatrixXf(_L.outerSize(), _L.outerSize());
    Eigen::SparseMatrix<float> L = *_L;
    CliqueSampler* sampler = new WeightedCliqueSampler();
    for(int i = 0; i < L.outerSize()-1; i++){
        // NOTE: The condition is an inequality instead of a strict equality to 0 to prevent
        // numerical instability
        if(L.coeff(i,i) >= 1e-8){
            _G->col(i) = L.col(i)/std::sqrt(L.coeff(i,i));
            Eigen::SparseMatrix<float> star = *getStar(&L, i);
            Eigen::MatrixXf approx_clique = sampler->sampleClique(L, i);
            // Eigen::MatrixXf exact_clique = star - L.col(i)*(L.row(i)/L.coeff(i,i));

            // std::cout << "Exact - approx clique = " << (exact_clique - approx_clique).norm() << std::endl;
            L = L - star + approx_clique;
        }
    }
}


Eigen::SparseMatrix<float>* NaiveApproximateCholesky::getStar(Eigen::SparseMatrix<float>* L, int i){

    Eigen::SparseMatrix<float> E_i(L->rows(), L->cols());
    E_i.reserve(Eigen::VectorXi::Constant(L->cols(), 2));
    for(int j = 0; j < L->cols(); j++){
        if(_L->coeff(i, j) != 0 && i!=j){
            float w = L->coeff(i, j);
            float sign = w > 0 ? 1.0 : -1.0;
            E_i.insert(i, j) = 1 * -sign * std::sqrt(std::abs(L->coeff(i, j)));
            E_i.insert(j, j) = -1 * std::sqrt(std::abs(L->coeff(i, j)));
        }
    }
    E_i.makeCompressed();
    Eigen::SparseMatrix<float>* star_i = new Eigen::SparseMatrix<float>(L->rows(), L->cols());
    *star_i = E_i * E_i.transpose();

    return star_i;
}

Eigen::SparseMatrix<float> NaiveApproximateCholesky::getLaplacian(){
    return *_L;
}

Eigen::MatrixXf NaiveApproximateCholesky::getReconstructedLaplacian(){
    return (*_G) * (_G->transpose());
}

// ApproximateCholesky

ApproximateCholesky::ApproximateCholesky(Eigen::SparseMatrix<float>* Adj){
    _A = Adj;
    _L = this->computeLaplacian(_A);
}

ApproximateCholesky::ApproximateCholesky(Eigen::SparseMatrix<float> Adj){
    _A = &Adj;
    _L = this->computeLaplacian(_A);
}

ApproximateCholesky::ApproximateCholesky(std::string filename, int nrows, int ncols){
    Reader* r = new TSVReader(filename, nrows, ncols);
    _A = r->Read();
    _L = this->computeLaplacian(_A);
}

Eigen::SparseMatrix<float> ApproximateCholesky::getAdjacencyMatrix(){
    return *_A;
}

Eigen::VectorXf ApproximateCholesky::solve(Eigen::VectorXf b){
    this->compute();
    PConjugateGradient pcg = PConjugateGradient(_L, &b);
    pcg.setPreconditioner(_ldli);
    Eigen::VectorXf x = pcg.solve(1e-12);
    return x;
}

void ApproximateCholesky::compute(){
    OrderedPreconditioner* prec = new OrderedPreconditioner(_A);
    _ldli = prec->getLDLi();
}

LDLi* ApproximateCholesky::getPreconditioner(){
    return _ldli;
}

Eigen::SparseMatrix<float> ApproximateCholesky::getLaplacian(){
    return *_L;
}

Eigen::MatrixXf ApproximateCholesky::getReconstructedLaplacian(){
    // TODO(kvignesh1420): Placeholder method for linking
    return *_L;
}
