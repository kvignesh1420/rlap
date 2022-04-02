#include <iostream>
#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/Eigen/SparseCholesky"
#include "rlap/cc/reader.h"
#include "rlap/cc/factorizers.h"
#include "rlap/cc/cg.h"
#include "rlap/cc/types.h"

#define BLOCK_SIZE 125000

Eigen::SparseMatrix<float>* getAdjacencyMatrix(std::string filepath, int nrows, int ncols){
    Reader* r = new TSVReader(filepath, nrows, ncols);
    Eigen::SparseMatrix<float>* A = r->Read();
    // sub-matrix matrix for faster tests
    int block_nrows = BLOCK_SIZE;
    int block_ncols = BLOCK_SIZE;
    int start_row = 0;
    int start_col = 0;
    Eigen::SparseMatrix<float>* A_block = new Eigen::SparseMatrix<float>(
        A->block(start_row, start_col, block_nrows, block_ncols)
    );
    return A_block;
}

int main(){

    int N = 125000;
    std::string filepath = "data/grid50.tsv";

    Eigen::SparseMatrix<float>* A = getAdjacencyMatrix(filepath, N, N);
    ApproximateCholesky fact = ApproximateCholesky(A);
    fact.compute();
    Eigen::SparseMatrix<float> L = fact.getLaplacian();
    LDLi* ldli = fact.getPreconditioner();

    Eigen::VectorXf x(BLOCK_SIZE);
    x.setZero();
    // set a random ground truth
	Eigen::VectorXf x_t = Eigen::VectorXf::Random(BLOCK_SIZE);
	// generate target based on random x_t
	Eigen::VectorXf b = L * x_t;
    // An alternative way is to directly generate the b vector
    // Eigen::VectorXf b = Eigen::VectorXf::Random(BLOCK_SIZE);

    float b_mean = b.mean();
    Eigen::VectorXf b_m = b - Eigen::VectorXf::Ones(b.size())*b_mean;
    // std::cout << " b_m.norm() = " << b_m.norm() << std::endl;

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

    return 0;
}
