#include <iostream>
#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/Eigen/SparseCholesky"
#include "rlap/cc/reader.h"
#include "rlap/cc/factorizers.h"
#include "rlap/cc/cg.h"
#include "rlap/cc/types.h"
#include "rlap/cc/preconditioner.h"

#define BLOCK_SIZE 10

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

    int N = 10;
    std::string filepath = "data/connected10.tsv";

    Eigen::SparseMatrix<float>* A = getAdjacencyMatrix(filepath, N, N);
    PriorityPreconditioner* prec = new PriorityPreconditioner(A);
    LLMatp* pmat = prec->getLLMatp();
    for(int i = 0; i < N; i++){
        prec->printColumn(pmat, i);
        std::cout << std::endl;
    }
    
    return 0;
}
