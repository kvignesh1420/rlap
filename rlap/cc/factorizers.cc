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
    OrderedMatrix* llmo = getOrderedMatrix();
    LDLi* ldli = computeLDLi(llmo);
    _ldli = ldli;
}

OrderedMatrix* ApproximateCholesky::getOrderedMatrix(){
    int n = _A->rows();
    std::vector<float> cols;
    std::vector<OrderedElement*> llelems;
    int ptr = 0;

    for(int i = 0; i < n-1; i++){
        float next = -1;
        int start_idx = _A->outerIndexPtr()[i];
        int end_idx = _A->outerIndexPtr()[i+1];
        for(int ind = start_idx ; ind < end_idx; ind++){
            int j = _A->innerIndexPtr()[ind];
            if(i < j){

                float v = _A->valuePtr()[ind];
                // v is value of the element at position (j,i) in _A.
                // std::cout << "_A(" << j << "," << i << ") = " << v << std::endl;
                OrderedElement* ord = new OrderedElement();
                ord->row = j;
                ord->next = next;
                ord->val = v;
                // llelems[ptr] = ord;
                llelems.push_back(ord);
                next = ptr;
                ptr += 1;
            }
        }
        cols.push_back(next);
    }

    OrderedMatrix* llmo = new OrderedMatrix();
    llmo->n = n;
    llmo->cols = cols;
    llmo->elements = llelems;
    return llmo;
}

float ApproximateCholesky::getColumnLength(OrderedMatrix* ordmat, int i, std::vector<ColumnElement>* colspace){
    // std::cout << "get col length" << std::endl;
    float ptr = ordmat->cols[i];
    // std::cout << "current ptr = " << ptr << std::endl;
    float len = 0;
    while(ptr != PTR_RESET){
        len += 1;
        ColumnElement item = ColumnElement();
        item.row = ordmat->elements[ptr]->row;
        item.ptr = ptr;
        item.val = ordmat->elements[ptr]->val;

        // std::cout << "Size of colspace = " << colspace->size() << std::endl;
        // std::cout << " row " << item.row << " ptr " << item.ptr <<  " value : " << item.val << std::endl;
        if(len > colspace->size()){
            colspace->push_back(item);
        }
        else{
            colspace->at(len-1) = item;
        }
        ptr = ordmat->elements[ptr]->next;
    }
    return len;
}

void ApproximateCholesky::printColumn(OrderedMatrix* ordmat, int i){
    float ptr = ordmat->cols[i];
    std::cout << "printing col " << i << std::endl;
    while(ptr != PTR_RESET){
        OrderedElement* oe = ordmat->elements[ptr];
        std::cout << "col " << i << " row " << oe->row << " value : " << oe->val << std::endl;
        ptr = oe->next;
    }
}

void ApproximateCholesky::printColspace(std::vector<ColumnElement>* colspace){
    std::cout << "COLSPACE = \n" << std::endl;
    for(int i = 0; i < colspace->size(); i++){
        ColumnElement item = colspace->at(i);
        std::cout << " row " << item.row << " ptr " << item.ptr <<  " value : " << item.val << std::endl;
    }
}

void ApproximateCholesky::printMatrixDetails(OrderedMatrix* ordmat){
    std::cout << " ======== Matrix Details =======" << std::endl;
    std::cout << "========= elements ===========" << std::endl;
    for(int i = 0; i < ordmat->elements.size(); i++){
        OrderedElement* oe = ordmat->elements[i];
        std::cout << " row " << oe->row << " next " << oe->next << " value : " << oe->val << std::endl;
    }
    std::cout << "========= COLS ===========" << std::endl;
    for(int i = 0; i < ordmat->cols.size(); i++){
        std::cout << "cols[" << i << "] = " << ordmat->cols[i] << std::endl;
    }
}

float ApproximateCholesky::compressColumn(std::vector<ColumnElement>* colspace, float len){
    // std::cout << "Compressing col" << std::endl;

    std::sort(colspace->begin(), colspace->begin()+len,
         [](ColumnElement a, ColumnElement b) {return a.row < b.row; });

    float ptr = PTR_RESET;
    float currow = colspace->at(0).row;
    float curptr = colspace->at(0).ptr;
    float curval = colspace->at(0).val;

    for(int i = 1; i < len; i++){
        if(colspace->at(i).row != currow){
            // std::cout << "currrow != colspace[i].row " << currow << " " << colspace->at(i).row << std::endl;
            ptr += 1;
            ColumnElement item = ColumnElement();
            item.row = currow;
            item.ptr = curptr;
            item.val = curval;
            colspace->at(ptr) = item;

            currow = colspace->at(i).row;
            curptr = colspace->at(i).ptr;
            curval = colspace->at(i).val;
        }
        else{
            // std::cout << "currrow == colspace[i].row " << currow << std::endl;
            curval += colspace->at(i).val;
        }
    }
    ptr += 1;
    ColumnElement item = ColumnElement();
    item.row = currow;
    item.ptr = curptr;
    item.val = curval;
    colspace->at(ptr) = item;
    std::sort(colspace->begin(), colspace->begin()+ptr+1,
         [](ColumnElement a, ColumnElement b) {return a.val < b.val; });

    return ptr+1;
}

LDLi* ApproximateCholesky::computeLDLi(OrderedMatrix* ordmat){
    float n = ordmat->n;
    LDLi* ldli = new LDLi();
    ldli->col = std::vector<float>();
    ldli->colptr = std::vector<float>();
    ldli->rowval = std::vector<float>();
    ldli->fval = std::vector<float>();

    float ldli_row_ptr = 0;
    std::vector<float> d(n, 0.0);
    float joffsets = 0;
    std::vector<ColumnElement>* colspace = new std::vector<ColumnElement>();
    std::mt19937_64 rand_generator;
    std::uniform_real_distribution<float> u_distribution(0, 1);
    for(int i = 0; i < n-1; i++){
        // std::cout << "====================== ComputeLDLi: column " << i << " ============================" << std::endl;

        ldli->col.push_back(i);
        ldli->colptr.push_back(ldli_row_ptr);

        float len = getColumnLength(ordmat, i, colspace);
        // std::cout << "Length from column with multiedges = " << len << std::endl;
        len = compressColumn(colspace, len);
        joffsets += len;
        // std::cout << "Length of column after compression = " << len << std::endl;

        float csum = 0;
        std::vector<float> cumspace;
        for(int ii = 0; ii < len ; ii++){
            csum += colspace->at(ii).val;
            cumspace.push_back(csum);
        }
        float wdeg = csum;
        float colScale = 1;

        for(int joffset = 0; joffset < len-1; joffset++){
            ColumnElement ColumnElement = colspace->at(joffset);
            float w = ColumnElement.val*colScale;
            float j = ColumnElement.row;
            float f = float(w)/wdeg;

            float u_r = u_distribution(rand_generator);
            float r = u_r*(csum  - cumspace[joffset]) + cumspace[joffset];
            float koff=-1;
            for(int k_i = 0; k_i < len; k_i++){
                if(cumspace[k_i]>r){
                    koff = k_i;
                    break;
                }
            }
            float k = colspace->at(koff).row;
            // std::cout << "random value r = "<< r << " current row = " << j << " satisfied row = " << k << std::endl;
            float newEdgeVal = w*(1-f);
            if(j < k){
                float jhead = ordmat->cols[j];

                OrderedElement* ord = new OrderedElement();
                ord->row = k;
                ord->next = jhead;
                ord->val = newEdgeVal;
                ordmat->elements[ColumnElement.ptr] = ord;

                ordmat->cols[j] = ColumnElement.ptr;
            } else{
                float khead = ordmat->cols[k];

                OrderedElement* ord = new OrderedElement();
                ord->row = j;
                ord->next = khead;
                ord->val = newEdgeVal;
                ordmat->elements[ColumnElement.ptr] = ord;

                ordmat->cols[k] = ColumnElement.ptr;
            }

            colScale = colScale*(1-f);
            wdeg = wdeg - 2*w + w*w/wdeg;

            ldli->rowval.push_back(j);
            ldli->fval.push_back(f);
            ldli_row_ptr += 1;
        }

        ColumnElement ColumnElement = colspace->at(len-1);
        float w = ColumnElement.val*colScale;
        float j = ColumnElement.row;
        ldli->rowval.push_back(j);
        ldli->fval.push_back(1.0);
        ldli_row_ptr += 1;

        d[i] = w;
    }

    ldli->colptr.push_back(ldli_row_ptr);
    ldli->d = d;
    // std::cout << " J OFFSETS = " << joffsets << std::endl;

    return ldli;
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
