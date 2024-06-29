#include <iostream>
#include <algorithm>
#include <vector>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <Eigen/SparseCore>
#include "preconditioner.h"
#include "types.h"
#include "tracer.h"

#define PTR_RESET -1

// PriorityPreconditioner

PriorityPreconditioner::PriorityPreconditioner(Eigen::SparseMatrix<double>* A, std::string o_n){
    _A = A;
    _o_n_str = o_n;
}

std::vector<double> PriorityPreconditioner::getFlipIndices(Eigen::SparseMatrix<double>* M){
    Eigen::SparseMatrix<double> F = Eigen::SparseMatrix<double>(M->rows(), M->cols());
    std::vector<Eigen::Triplet<double> > triplets;

    int64_t n = M->rows();
    double counter = 0;
    for(int64_t coln = 0; coln < n; coln++){
        int64_t start_idx = M->outerIndexPtr()[coln];
        int64_t end_idx = M->outerIndexPtr()[coln+1];
        for(int64_t ind = start_idx ; ind < end_idx; ind++){
            int64_t rown  = M->innerIndexPtr()[ind];
            triplets.push_back(Eigen::Triplet<double>(rown, coln, counter));
            counter += 1;
        }
    }
    F.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SparseMatrix<double> F_t = F.transpose();
    std::vector<double> flipped_indices;
    for(int64_t coln = 0; coln < n; coln++){
        int64_t start_idx = F_t.outerIndexPtr()[coln];
        int64_t end_idx = F_t.outerIndexPtr()[coln+1];
        for(int64_t ind = start_idx ; ind < end_idx; ind++){
            flipped_indices.push_back(F_t.valuePtr()[ind]);
        }
    }
    return flipped_indices;
}

void PriorityPreconditioner::printFlipIndices(std::vector<double> fi){
    std::cout << "flip indices = " << std::endl;
    for(int64_t i = 0; i < fi.size(); i++){
        double val = fi.at(i);
        if(i != fi.at(val)){
            std::cout << "flip index mismatch: " << i << " " << val << std::endl;
        }
        else{
            std::cout << "original index: " << i << " flipped index: " << fi.at(i) << " " << std::endl;
        }
    }
    std::cout << std::endl;
}

PriorityMatrix* PriorityPreconditioner::getPriorityMatrix(){

    int64_t n = _A->rows();
    std::vector<PriorityElement*> cols;
    std::vector<PriorityElement*> llelems;
    std::vector<double> degs;
    std::vector<double> flips = getFlipIndices(_A);

    for(int64_t i = 0; i < n; i++){
        int64_t start_idx = _A->outerIndexPtr()[i];
        int64_t end_idx = _A->outerIndexPtr()[i+1];
        double deg = end_idx - start_idx;
        degs.push_back(deg);
        if(deg == 0){
            PriorityElement* pe = new PriorityElement();
            cols.push_back(pe);
            continue;
        }

        double j = _A->innerIndexPtr()[start_idx];
        double v = _A->valuePtr()[start_idx];
        PriorityElement* pe = new PriorityElement(j, v);
        llelems.push_back(pe);
        PriorityElement* next = pe;

        for(int64_t ind = start_idx + 1; ind < end_idx; ind++){
            j = _A->innerIndexPtr()[ind];
            v = _A->valuePtr()[ind];
            PriorityElement* pe = new PriorityElement(j, v, next);
            llelems.push_back(pe);
            next = pe;
        }
        cols.push_back(next);
    }

    for(int64_t i = 0; i < n; i++){
        int64_t start_idx = _A->outerIndexPtr()[i];
        int64_t end_idx = _A->outerIndexPtr()[i+1];
        for(int64_t ind = start_idx; ind < end_idx; ind++){
            llelems.at(ind)->reverse = llelems.at(flips.at(ind));
        }
    }

    PriorityMatrix* pmat = new PriorityMatrix();
    pmat->n = n;
    pmat->degs = degs;
    pmat->cols = cols;
    pmat->lles = llelems;
    return pmat;
}

void PriorityPreconditioner::printColumn(PriorityMatrix* pmat, int64_t i){
    PriorityElement* ll = pmat->cols.at(i);
    std::cout << "col " << i << " row " << ll->row << " value " << ll->val << std::endl;
    while(ll->next!=ll){
        ll = ll->next;
        std::cout << "col " << i << " row " << ll->row << " value " << ll->val << std::endl;
    }
}

DegreePQ* PriorityPreconditioner::getDegreePQ(std::vector<double> degs){
    double n = degs.size();
    std::vector<DegreePQElement*> elems;
    for(int64_t i = 0; i < n; i++){
        elems.push_back(nullptr);
    }
    std::vector<double> lists;
    for(int64_t i = 0; i < 2*n + 1; i++){
        lists.push_back(-1);
    }
    double minlist = 0;

    for(int64_t i = 0; i < n; i++){
        double key = degs.at(i);
        double head = lists.at(key);

        if(head >= 0){
            DegreePQElement* elem_i = new DegreePQElement();
            elem_i->prev = -1;
            elem_i->next = head;
            elem_i->key = key;
            elems.at(i) = elem_i;
            elems.at(head)->prev = i;
        }
        else{
            DegreePQElement* elem_i = new DegreePQElement();
            elem_i->prev = -1;
            elem_i->next = -1;
            elem_i->key = key;
            elems.at(i) = elem_i;
        }
        lists.at(key) = i;
    }
    DegreePQ* pq = new DegreePQ();
    pq->elems = elems;
    pq->lists = lists;
    pq->minlist = minlist;
    pq->nitems = n;
    pq->n = n;
    return pq;
}

double PriorityPreconditioner::DegreePQPop(DegreePQ* pq){
    if(pq->nitems == 0){
        throw std::invalid_argument("the PQ is empty. Cannot pop an element");
    }
    while (pq->lists.at(pq->minlist) == -1){
        pq->minlist += 1;
    }
    // pq->minlist now represents the min degree of the nodes in the graph
    double i = pq->lists.at(pq->minlist);
    // i is the node with min degree and will be popped
    double next = pq->elems.at(i)->next;
    // update the index of the node with min-degree in pq->lists
    pq->lists.at(pq->minlist) = next;

    delete pq->elems.at(i);
    pq->elems.at(i) = nullptr;
    if(next > -1){
        pq->elems.at(next)->prev = -1;
    }

    pq->nitems -= 1;

    return i;
}

void PriorityPreconditioner::DegreePQMove(DegreePQ* pq, int64_t i, double newkey, int64_t oldlist, int64_t newlist){

    double prev = pq->elems.at(i)->prev;
    double next = pq->elems.at(i)->next;
    if(next > -1){
        pq->elems.at(next)->prev = prev;
    }
    if(prev > -1){
        pq->elems.at(prev)->next = next;
    } else{
        pq->lists.at(oldlist) = next;
    }

    double head = pq->lists.at(newlist);
    if(head > -1){
        pq->elems.at(head)->prev = i;
    }
    pq->lists.at(newlist) = i;
    pq->elems.at(i)->prev = -1;
    pq->elems.at(i)->next = head;
    pq->elems.at(i)->key = newkey;
    return;
}

void PriorityPreconditioner::DegreePQDec(DegreePQ* pq, int64_t i){
    double n = pq->n;
    double deg_i = pq->elems.at(i)->key;
    if(deg_i == 1){
        return;
    }
    int64_t oldlist = deg_i <= n ? deg_i : n + int(deg_i/n);
    int64_t newlist = deg_i-1 <= n ? deg_i-1 : n + int((deg_i-1)/n);
    if(oldlist != newlist){
        DegreePQMove(pq, i, deg_i-1, oldlist, newlist);
        if(newlist < pq->minlist){
            pq->minlist = newlist;
        }
    }else{
        pq->elems.at(i)->key -= 1;
    }
    return;
}

void PriorityPreconditioner::DegreePQInc(DegreePQ* pq, int64_t i){
    double n = pq->n;
    double deg_i = pq->elems.at(i)->key;
    int64_t oldlist = deg_i <= n ? deg_i : n + int(deg_i/n);
    int64_t newlist = deg_i+1 <= n ? deg_i+1 : n + int((deg_i+1)/n);
    if(oldlist != newlist){
        DegreePQMove(pq, i, deg_i+1, oldlist, newlist);
    }else{
        pq->elems.at(i)->key += 1;
    }
    return;
}

double PriorityPreconditioner::getColumnLength(PriorityMatrix* pmat, int64_t i, std::vector<PriorityElement*>* colspace){
    PriorityElement* ll = pmat->cols[i];
    double len = 0;
    while(ll->next != ll){
        if(ll->val > 0){
            len += 1;
            if(len > colspace->size()){
                colspace->push_back(ll);
            } else{
                colspace->at(len-1) = ll;
            }
        }
        ll = ll->next;
    }
    if(ll->val > 0){
        len += 1;
        if(len > colspace->size()){
            colspace->push_back(ll);
        } else{
            colspace->at(len-1) = ll;
        }
    }
    return len;
}

double PriorityPreconditioner::compressColumn(std::vector<PriorityElement*>* colspace, double len, DegreePQ* pq){

    std::sort(colspace->begin(), colspace->begin()+len,
         [](PriorityElement* j, PriorityElement* k) {return j->row < k->row; });

    double ptr = PTR_RESET;
    double currow = -1;

    for(int64_t i = 0; i < len; i++){
        if(colspace->at(i)->row != currow){
            currow = colspace->at(i)->row;
            ptr += 1;
            colspace->at(ptr) = colspace->at(i);
        }
        else{
            colspace->at(ptr)->val += colspace->at(i)->val;
            colspace->at(i)->reverse->val = 0;

            DegreePQDec(pq, currow);
        }
    }

    if (_o_n_str == "asc"){
        std::sort(colspace->begin(), colspace->begin()+ptr+1,
         [](PriorityElement* j, PriorityElement* k) {return j->val < k->val; });
    }
    else if (_o_n_str == "desc"){
        std::sort(colspace->begin(), colspace->begin()+ptr+1,
         [](PriorityElement* j, PriorityElement* k) {return j->val > k->val; });
    }
    else if (_o_n_str == "random"){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle ( colspace->begin(), colspace->begin()+ptr+1, gen );
    }

    return ptr+1;
}

double PriorityPreconditioner::compressColumnSC(std::vector<PriorityElement*>* colspace, double len, DegreePQ* pq){

    std::sort(colspace->begin(), colspace->begin()+len,
         [](PriorityElement* j, PriorityElement* k) {return j->row < k->row; });

    double ptr = PTR_RESET;
    double currow = -1;

    for(int64_t i = 0; i < len; i++){
        if(colspace->at(i)->row != currow){
            currow = colspace->at(i)->row;
            ptr += 1;
            colspace->at(ptr) = colspace->at(i);
        }
        else{
            colspace->at(ptr)->val += colspace->at(i)->val;
        }
    }

    if (_o_n_str == "asc"){
        std::sort(colspace->begin(), colspace->begin()+ptr+1,
         [](PriorityElement* j, PriorityElement* k) {return j->val < k->val; });
    }
    else if (_o_n_str == "desc"){
        std::sort(colspace->begin(), colspace->begin()+ptr+1,
         [](PriorityElement* j, PriorityElement* k) {return j->val > k->val; });
    }
    else if (_o_n_str == "random"){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle ( colspace->begin(), colspace->begin()+ptr+1, gen );
    }
    return ptr+1;
}


Eigen::MatrixXd PriorityPreconditioner::getSchurComplement(int64_t t){
    PriorityMatrix* a = getPriorityMatrix();
    double n = a->n;

    DegreePQ* pq = getDegreePQ(a->degs);

    double it = 1;
    std::vector<PriorityElement*>* colspace = new std::vector<PriorityElement*>();
    std::mt19937_64 rand_generator;
    std::uniform_real_distribution<double> u_distribution(0, 1);
    while(it <= t && it < n){

        double i = DegreePQPop(pq);

        it += 1;

        double len = getColumnLength(a, i, colspace);
        len = compressColumn(colspace, len, pq);
        double csum = 0;
        std::vector<double> cumspace;
        std::vector<double> vals;
        for(int64_t ii = 0; ii < len ; ii++){
            vals.push_back(colspace->at(ii)->val);
            csum += colspace->at(ii)->val;
            cumspace.push_back(csum);
        }
        double wdeg = csum;
        double colScale = 1;

        for(int64_t joffset = 0; joffset < len-1; joffset++){
            PriorityElement* ll = colspace->at(joffset);
            double w = vals.at(joffset) * colScale;
            double j = ll->row;
            PriorityElement* revj = ll->reverse;
            double f = double(w)/wdeg;
            vals.at(joffset) = 0;

            double u_r = u_distribution(rand_generator);
            double r = u_r*(csum  - cumspace[joffset]) + cumspace[joffset];
            double koff = len-1;
            for(int64_t k_i = 0; k_i < len; k_i++){
                if(cumspace[k_i]>r){
                    koff = k_i;
                    break;
                }
            }
            double k = colspace->at(koff)->row;

            // A new edge is being added from node j to node k.
            // However, since the edge between node i and node j is being removed
            // the degree of node j is not incremented.
            DegreePQInc(pq, k);

            double newEdgeVal = f*(1-f)*wdeg;

            // row k in col j
            revj->row = k;
            revj->val = newEdgeVal;
            revj->reverse = ll;

            // row j in col k
            PriorityElement* khead = a->cols.at(k);
            a->cols.at(k) = ll;
            ll->next = khead;
            ll->reverse = revj;
            ll->val = newEdgeVal;
            ll->row = j;

            colScale = colScale*(1-f);
            wdeg = wdeg*(1-f)*(1-f);
        }

        if(len > 0){
            PriorityElement* ll = colspace->at(len-1);
            double j = ll->row;
            PriorityElement* revj = ll->reverse;
            if(it < n){
                // This indicates that the edge from node i to node j
                // has been removed
                DegreePQDec(pq, j);
            }
            ll->val = 0;
            revj->val = 0;

        }
    }

    std::vector<Eigen::Vector3d> edge_info_vector;
    Eigen::MatrixXd edge_info;


    int64_t edge_info_counter = 0;
    while (pq->nitems > 0){
        double i = DegreePQPop(pq);
        double len = getColumnLength(a, i, colspace);

        len = compressColumnSC(colspace, len, pq);

        for(int64_t ii = 0; ii < len ; ii++){
            double val = colspace->at(ii)->val;
            double row = colspace->at(ii)->row;
            Eigen::Vector3d temp(row, i, val);
            edge_info_vector.push_back(temp);
            edge_info_counter += 1;
        }
    }
    edge_info.resize(edge_info_counter, 3);
    for(int64_t z = 0; z < edge_info_counter; z++){
        edge_info.row(z) = edge_info_vector[z];
    }

    delete colspace;
    colspace = nullptr;

    for(auto pq_element: pq->elems){
        delete pq_element;
        pq_element = nullptr;
    }
    delete pq;
    pq = nullptr;

    for(auto ll_element: a->lles){
        delete ll_element;
        ll_element = nullptr;
    }
    delete a;
    a = nullptr;
    return edge_info;
}

// RandomPreconditioner


RandomPreconditioner::RandomPreconditioner(Eigen::SparseMatrix<double>* A, std::string o_n){
    _A = A;
    _o_n_str = o_n;
}

std::vector<double> RandomPreconditioner::getFlipIndices(Eigen::SparseMatrix<double>* M){
    Eigen::SparseMatrix<double> F = Eigen::SparseMatrix<double>(M->rows(), M->cols());
    std::vector<Eigen::Triplet<double> > triplets;

    int64_t n = M->rows();
    double counter = 0;
    for(int64_t coln = 0; coln < n; coln++){
        int64_t start_idx = M->outerIndexPtr()[coln];
        int64_t end_idx = M->outerIndexPtr()[coln+1];
        for(int64_t ind = start_idx ; ind < end_idx; ind++){
            int64_t rown  = M->innerIndexPtr()[ind];
            triplets.push_back(Eigen::Triplet<double>(rown, coln, counter));
            counter += 1;
        }
    }
    F.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::SparseMatrix<double> F_t = F.transpose();
    std::vector<double> flipped_indices;
    for(int64_t coln = 0; coln < n; coln++){
        int64_t start_idx = F_t.outerIndexPtr()[coln];
        int64_t end_idx = F_t.outerIndexPtr()[coln+1];
        for(int64_t ind = start_idx ; ind < end_idx; ind++){
            flipped_indices.push_back(F_t.valuePtr()[ind]);
        }
    }
    return flipped_indices;
}

void RandomPreconditioner::printFlipIndices(std::vector<double> fi){
    std::cout << "flip indices = " << std::endl;
    for(int64_t i = 0; i < fi.size(); i++){
        double val = fi.at(i);
        if(i != fi.at(val)){
            std::cout << "flip index mismatch: " << i << " " << val << std::endl;
        }
        else{
            std::cout << "original index: " << i << " flipped index: " << fi.at(i) << " " << std::endl;
        }
    }
    std::cout << std::endl;
}

PriorityMatrix* RandomPreconditioner::getPriorityMatrix(){

    int64_t n = _A->rows();
    std::vector<PriorityElement*> cols;
    std::vector<PriorityElement*> llelems;
    std::vector<double> degs;
    std::vector<double> flips = getFlipIndices(_A);

    for(int64_t i = 0; i < n; i++){
        int64_t start_idx = _A->outerIndexPtr()[i];
        int64_t end_idx = _A->outerIndexPtr()[i+1];
        double deg = end_idx - start_idx;
        degs.push_back(deg);
        if(deg == 0){
            PriorityElement* pe = new PriorityElement();
            cols.push_back(pe);
            continue;
        }

        double j = _A->innerIndexPtr()[start_idx];
        double v = _A->valuePtr()[start_idx];
        PriorityElement* pe = new PriorityElement(j, v);
        llelems.push_back(pe);
        PriorityElement* next = pe;

        for(int64_t ind = start_idx + 1; ind < end_idx; ind++){
            j = _A->innerIndexPtr()[ind];
            v = _A->valuePtr()[ind];
            PriorityElement* pe = new PriorityElement(j, v, next);
            llelems.push_back(pe);
            next = pe;
        }
        cols.push_back(next);
    }

    for(int64_t i = 0; i < n; i++){
        int64_t start_idx = _A->outerIndexPtr()[i];
        int64_t end_idx = _A->outerIndexPtr()[i+1];
        for(int64_t ind = start_idx; ind < end_idx; ind++){
            llelems.at(ind)->reverse = llelems.at(flips.at(ind));
        }
    }

    PriorityMatrix* pmat = new PriorityMatrix();
    pmat->n = n;
    pmat->degs = degs;
    pmat->cols = cols;
    pmat->lles = llelems;
    return pmat;
}

void RandomPreconditioner::printColumn(PriorityMatrix* pmat, int64_t i){
    PriorityElement* ll = pmat->cols.at(i);
    std::cout << "col " << i << " row " << ll->row << " value " << ll->val << std::endl;
    while(ll->next!=ll){
        ll = ll->next;
        std::cout << "col " << i << " row " << ll->row << " value " << ll->val << std::endl;
    }
}

RandomPQ* RandomPreconditioner::getRandomPQ(std::vector<double> degs){
    double n = degs.size();
    std::vector<double> node_id;
    for(int64_t i = 0; i < n; i++){
        node_id.push_back(i);
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle ( node_id.begin(), node_id.end(), gen );
    RandomPQ* pq = new RandomPQ();
    pq->node_id = node_id;
    pq->nitems = n;
    return pq;
}

double RandomPreconditioner::RandomPQPop(RandomPQ* pq){

    if(pq->nitems == 0){
        throw std::invalid_argument("the PQ is empty. Cannot pop an element");
    }

    double i = pq->node_id[pq->nitems - 1];
    pq->nitems -= 1;

    return i;
}


double RandomPreconditioner::getColumnLength(PriorityMatrix* pmat, int64_t i, std::vector<PriorityElement*>* colspace){
    PriorityElement* ll = pmat->cols[i];
    double len = 0;
    while(ll->next != ll){
        if(ll->val > 0){
            len += 1;
            if(len > colspace->size()){
                colspace->push_back(ll);
            } else{
                colspace->at(len-1) = ll;
            }
        }
        ll = ll->next;
    }
    if(ll->val > 0){
        len += 1;
        if(len > colspace->size()){
            colspace->push_back(ll);
        } else{
            colspace->at(len-1) = ll;
        }
    }
    return len;
}

double RandomPreconditioner::compressColumn(std::vector<PriorityElement*>* colspace, double len){

    std::sort(colspace->begin(), colspace->begin()+len,
         [](PriorityElement* j, PriorityElement* k) {return j->row < k->row; });

    double ptr = PTR_RESET;
    double currow = -1;

    for(int64_t i = 0; i < len; i++){
        if(colspace->at(i)->row != currow){

            currow = colspace->at(i)->row;
            ptr += 1;
            colspace->at(ptr) = colspace->at(i);
        }
        else{
            colspace->at(ptr)->val += colspace->at(i)->val;
            colspace->at(i)->reverse->val = 0;
        }
    }

    if (_o_n_str == "asc"){
        std::sort(colspace->begin(), colspace->begin()+ptr+1,
         [](PriorityElement* j, PriorityElement* k) {return j->val < k->val; });
    }
    else if (_o_n_str == "desc"){
        std::sort(colspace->begin(), colspace->begin()+ptr+1,
         [](PriorityElement* j, PriorityElement* k) {return j->val > k->val; });
    }
    else if (_o_n_str == "random"){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle ( colspace->begin(), colspace->begin()+ptr+1, gen );
    }
    return ptr+1;
}

double RandomPreconditioner::compressColumnSC(std::vector<PriorityElement*>* colspace, double len){

    std::sort(colspace->begin(), colspace->begin()+len,
         [](PriorityElement* j, PriorityElement* k) {return j->row < k->row; });

    double ptr = PTR_RESET;
    double currow = -1;

    for(int64_t i = 0; i < len; i++){
        if(colspace->at(i)->row != currow){
            currow = colspace->at(i)->row;
            ptr += 1;
            colspace->at(ptr) = colspace->at(i);
        }
        else{
            colspace->at(ptr)->val += colspace->at(i)->val;
        }
    }

    if (_o_n_str == "asc"){
        std::sort(colspace->begin(), colspace->begin()+ptr+1,
         [](PriorityElement* j, PriorityElement* k) {return j->val < k->val; });
    }
    else if (_o_n_str == "desc"){
        std::sort(colspace->begin(), colspace->begin()+ptr+1,
         [](PriorityElement* j, PriorityElement* k) {return j->val > k->val; });
    }
    else if (_o_n_str == "random"){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle ( colspace->begin(), colspace->begin()+ptr+1, gen );
    }
    return ptr+1;
}

Eigen::MatrixXd RandomPreconditioner::getSchurComplement(int64_t t){
    PriorityMatrix* a = getPriorityMatrix();
    double n = a->n;

    RandomPQ* pq = getRandomPQ(a->degs);

    double it = 1;
    std::vector<PriorityElement*>* colspace = new std::vector<PriorityElement*>();
    std::mt19937_64 rand_generator;
    std::uniform_real_distribution<double> u_distribution(0, 1);
    while(it <= t && it < n){

        double i = RandomPQPop(pq);

        double len = getColumnLength(a, i, colspace);
        len = compressColumn(colspace, len);
        double csum = 0;
        std::vector<double> cumspace;
        std::vector<double> vals;
        for(int64_t ii = 0; ii < len ; ii++){
            vals.push_back(colspace->at(ii)->val);
            csum += colspace->at(ii)->val;
            cumspace.push_back(csum);
        }
        double wdeg = csum;
        double colScale = 1;

        for(int64_t joffset = 0; joffset < len-1; joffset++){
            PriorityElement* ll = colspace->at(joffset);
            double w = vals.at(joffset) * colScale;
            double j = ll->row;
            PriorityElement* revj = ll->reverse;
            double f = double(w)/wdeg;
            vals.at(joffset) = 0;

            double u_r = u_distribution(rand_generator);
            double r = u_r*(csum  - cumspace[joffset]) + cumspace[joffset];
            double koff = len-1;
            for(int64_t k_i = 0; k_i < len; k_i++){
                if(cumspace[k_i]>r){
                    koff = k_i;
                    break;
                }
            }
            double k = colspace->at(koff)->row;

            double newEdgeVal = f*(1-f)*wdeg;

            // row k in col j
            revj->row = k;
            revj->val = newEdgeVal;
            revj->reverse = ll;

            // row j in col k
            PriorityElement* khead = a->cols.at(k);
            a->cols.at(k) = ll;
            ll->next = khead;
            ll->reverse = revj;
            ll->val = newEdgeVal;
            ll->row = j;

            colScale = colScale*(1-f);
            wdeg = wdeg*(1-f)*(1-f);
        }

        if(len > 0){
            PriorityElement* ll = colspace->at(len-1);
            PriorityElement* revj = ll->reverse;
            ll->val = 0;
            revj->val = 0;

        }

        it += 1;
    }

    std::vector<Eigen::Vector3d> edge_info_vector;
    Eigen::MatrixXd edge_info;

    int64_t edge_info_counter = 0;
    while (pq->nitems > 0){
        double i = RandomPQPop(pq);

        double len = getColumnLength(a, i, colspace);

        len = compressColumnSC(colspace, len);
        for(int64_t ii = 0; ii < len ; ii++){
            double val = colspace->at(ii)->val;
            double row = colspace->at(ii)->row;
            Eigen::Vector3d temp(row, i, val);
            edge_info_vector.push_back(temp);
            edge_info_counter += 1;
        }
    }
    edge_info.resize(edge_info_counter, 3);
    for(int64_t z = 0; z < edge_info_counter; z++){
        edge_info.row(z) = edge_info_vector[z];
    }

    delete colspace;
    colspace = nullptr;

    delete pq;
    pq = nullptr;

    for(auto ll_element: a->lles){
        delete ll_element;
        ll_element = nullptr;
    }
    delete a;
    a = nullptr;
    return edge_info;
}


// CoarseningPreconditioner

CoarseningPreconditioner::CoarseningPreconditioner(Eigen::SparseMatrix<double>* A):
    PriorityPreconditioner(A, "random"){
    _A = A;
}

Eigen::MatrixXd CoarseningPreconditioner::getSchurComplement(int64_t t){

    PriorityMatrix* a = getPriorityMatrix();
    double n = a->n;

    DegreePQ* pq = getDegreePQ(a->degs);

    double it = 1;
    std::vector<PriorityElement*>* colspace = new std::vector<PriorityElement*>();
    std::mt19937_64 rand_generator;
    std::uniform_real_distribution<double> u_distribution(0, 1);
    while(it <= t && it < n){
        double i = DegreePQPop(pq);

        it += 1;

        double len = getColumnLength(a, i, colspace);
        len = compressColumn(colspace, len, pq);
        // continue to the next node if there are no edges for this node
        if(len < 1) continue;

        double csum = 0;
        std::vector<double> cumspace;
        std::vector<double> vals;
        for(int64_t ii = 0; ii < len ; ii++){
            vals.push_back(colspace->at(ii)->val);
            csum += colspace->at(ii)->val;
            cumspace.push_back(csum);
        }

        // choose the neighbour with probability proportional
        // to its weight
        double u_r = u_distribution(rand_generator);
        double r = u_r*(csum);
        double koff = len-1;
        for(int64_t k_i = 0; k_i < len; k_i++){
            if(cumspace[k_i]>r){
                koff = k_i;
                break;
            }
        }
        double k = colspace->at(koff)->row;
        double w_k = vals.at(koff);
        PriorityElement* k_pe = colspace->at(koff);
        // collapse the current node onto the selected neighbour
        k_pe->val = 0;
        k_pe->reverse->val = 0;
        DegreePQDec(pq, k);
        // remove edges from node i to it's neighbours and
        // add random edges among it's neighbours from the selected
        // neighbour k
        for(int64_t joffset = 0; joffset < len; joffset++){
            if(joffset == koff) continue;

            PriorityElement* ll = colspace->at(joffset);

            double w = vals.at(joffset);
            double j = ll->row;

            PriorityElement* revj = ll->reverse;
            vals.at(joffset) = 0;
            DegreePQInc(pq, k);
            double newEdgeVal = (w_k * w)/(w_k + w);

            // row k in col j
            revj->row = k;
            revj->val = newEdgeVal;
            revj->reverse = ll;

            // row j in col k
            PriorityElement* khead = a->cols.at(k);
            a->cols.at(k) = ll;
            ll->next = khead;
            ll->reverse = revj;
            ll->val = newEdgeVal;
            ll->row = j;

        }

    }

    std::vector<Eigen::Vector3d> edge_info_vector;
    Eigen::MatrixXd edge_info;
    int64_t edge_info_counter = 0;
    while (pq->nitems > 0){
        double i = DegreePQPop(pq);
        double len = getColumnLength(a, i, colspace);
        len = compressColumnSC(colspace, len, pq);
        for(int64_t ii = 0; ii < len ; ii++){
            double val = colspace->at(ii)->val;
            double row = colspace->at(ii)->row;
            Eigen::Vector3d temp(row, i, val);
            edge_info_vector.push_back(temp);
            edge_info_counter += 1;
        }
    }
    edge_info.resize(edge_info_counter, 3);
    for(int64_t z = 0; z < edge_info_counter; z++){
        edge_info.row(z) = edge_info_vector[z];
    }

    // delete colspace
    delete colspace;
    colspace = nullptr;

    // delete left-over pointers to pq elements
    for(auto pq_element: pq->elems){
        delete pq_element;
        pq_element = nullptr;
    }
    delete pq;
    pq = nullptr;

    // delete the pointers in priority matrix
    for(auto ll_element: a->lles){
        delete ll_element;
        ll_element = nullptr;
    }
    delete a;
    a = nullptr;
    return edge_info;

}
