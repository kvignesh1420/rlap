#include <iostream>
#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/Eigen/SparseCholesky"
#include "rlap/cc/reader.h"
#include "rlap/cc/factorizers.h"
#include "rlap/cc/cg.h"
#include "rlap/cc/types.h"
#include "rlap/cc/preconditioner.h"

Eigen::SparseMatrix<double>* getAdjacencyMatrix(std::string filepath, int nrows, int ncols){
    Reader* r = new TSVReader(filepath, nrows, ncols);
    Eigen::SparseMatrix<double>* A = r->Read();
    return A;
}

int main(){

    int N = 125000;
    std::string filepath = "data/grid50.tsv";

    Eigen::SparseMatrix<double>* A = getAdjacencyMatrix(filepath, N, N);
    std::cout << "nnz(A) = " << A->nonZeros() << std::endl;
    // initialize the factorizer, the default preconditioner is
    // the DegreePreconditioner, other options include:
    // "order" for OrderedPreconditioner
    ApproximateCholesky fact = ApproximateCholesky(/* *Adj= */A, /*pre=*/"degree");

    // retrieve the computed laplacian
    Eigen::SparseMatrix<double> L = fact.getLaplacian();
    LDLi* ldli = fact.getPreconditioner();

    Eigen::VectorXd x(N);
    x.setZero();
    // set a random ground truth
	Eigen::VectorXd x_t = Eigen::VectorXd::Random(N);
	// generate target based on random x_t
	Eigen::VectorXd b = L * x_t;
    // An alternative way is to directly generate the b vector
    // Eigen::VectorXd b = Eigen::VectorXd::Random(N);

    double b_mean = b.mean();
    Eigen::VectorXd b_m = b - Eigen::VectorXd::Ones(b.size())*b_mean;
    std::cout << " b_m.norm() = " << b_m.norm() << std::endl;
    // observe the precision of the mean.
    std::cout << " b_m.mean() = " << b_m.mean() << std::endl;
    // exit(1);
    // pre-conditioning: NOTE than since the diagonal elements of G can be zero,
    // it is not possible to invert it and do a triangular solve that is
    // generally possible with positive definite matrices.

    // solve Lx = b_m with pre-conditioning
    std::cout << "initializing the PCG solver" << std::endl;
    PConjugateGradient pcg = PConjugateGradient(&L, &b_m);
    pcg.setPreconditioner(ldli);
    std::cout << "start solving using PCG" << std::endl;
    x = pcg.solve(1e-12);

    std::cout << "(L*x - b_m).norm()/b_m.norm() = " << (L*x - b_m).norm()/b_m.norm() << std::endl;
    std::cout << "abs(x_t.norm() - x.norm()) = " << std::abs(x_t.norm() - x.norm()) << std::endl;

    filepath.clear();
    return 0;
}
