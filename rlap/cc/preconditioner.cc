#include <iostream>
#include <algorithm>
#include <vector>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include "third_party/eigen3/Eigen/SparseCore"
#include "preconditioner.h"
#include "types.h"
#include "tracer.h"

#define PTR_RESET -1


// PriorityPreconditioner

PriorityPreconditioner::PriorityPreconditioner(Eigen::SparseMatrix<double>* A){
    _A = A;
}

std::vector<double> PriorityPreconditioner::getFlipIndices(Eigen::SparseMatrix<double>* M){
    Eigen::SparseMatrix<double> F = Eigen::SparseMatrix<double>(M->rows(), M->cols());
    std::vector<Eigen::Triplet<double> > triplets;

    int n = M->rows();
    double counter = 0;
    for(int coln = 0; coln < n; coln++){
        int start_idx = M->outerIndexPtr()[coln];
        int end_idx = M->outerIndexPtr()[coln+1];
        for(int ind = start_idx ; ind < end_idx; ind++){
            int rown  = M->innerIndexPtr()[ind];
            triplets.push_back(Eigen::Triplet<double>(rown, coln, counter));
            counter += 1;
        }
    }
    F.setFromTriplets(triplets.begin(), triplets.end());
    // std::cout << "F matrix = \n" << F << std::endl;
    Eigen::SparseMatrix<double> F_t = F.transpose();
    // std::cout << "F_t matrix = \n" << F_t << std::endl;
    std::vector<double> flipped_indices;
    for(int coln = 0; coln < n; coln++){
        int start_idx = F_t.outerIndexPtr()[coln];
        int end_idx = F_t.outerIndexPtr()[coln+1];
        for(int ind = start_idx ; ind < end_idx; ind++){
            flipped_indices.push_back(F_t.valuePtr()[ind]);
        }
    }
    return flipped_indices;
}

void PriorityPreconditioner::printFlipIndices(std::vector<double> fi){
    std::cout << "flip indices = " << std::endl;
    for(int i = 0; i < fi.size(); i++){
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
    // std::cout << "Getting PriorityMatrix " << std::endl;
    int n = _A->rows();
    std::vector<PriorityElement*> cols;
    std::vector<PriorityElement*> llelems;
    std::vector<double> degs;

    // std::cout << "flipping " << std::endl;
    std::vector<double> flips = getFlipIndices(_A);
    // printFlipIndices(flips);
    // std::cout << "flipped " << std::endl;

    for(int i = 0; i < n; i++){
        int start_idx = _A->outerIndexPtr()[i];
        int end_idx = _A->outerIndexPtr()[i+1];
        double deg = end_idx - start_idx;
        degs.push_back(deg);
        if(deg == 0){
            // std::cout << " deg[" << i << "] = " <<  end_idx - start_idx << std::endl;
            PriorityElement* pe = new PriorityElement();
            cols.push_back(pe);
            continue;
        }
        // std::cout << " deg[" << i << "] = " <<  end_idx - start_idx << std::endl;

        double j = _A->innerIndexPtr()[start_idx];
        double v = _A->valuePtr()[start_idx];
        PriorityElement* pe = new PriorityElement(j, v);
        llelems.push_back(pe);
        PriorityElement* next = pe;

        for(int ind = start_idx + 1; ind < end_idx; ind++){
            j = _A->innerIndexPtr()[ind];
            v = _A->valuePtr()[ind];
            PriorityElement* pe = new PriorityElement(j, v, next);
            llelems.push_back(pe);
            next = pe;
        }
        cols.push_back(next);
    }

    for(int i = 0; i < n; i++){
        int start_idx = _A->outerIndexPtr()[i];
        int end_idx = _A->outerIndexPtr()[i+1];
        for(int ind = start_idx; ind < end_idx; ind++){
            llelems.at(ind)->reverse = llelems.at(flips.at(ind));
        }
    }

    PriorityMatrix* pmat = new PriorityMatrix();
    pmat->n = n;
    pmat->degs = degs;
    pmat->cols = cols;
    pmat->lles = llelems;
    // std::cout << "Got LLMAtp " << "degs size = " << degs.size() << std::endl;
    return pmat;
}

void PriorityPreconditioner::printColumn(PriorityMatrix* pmat, int i){
    PriorityElement* ll = pmat->cols.at(i);
    std::cout << "col " << i << " row " << ll->row << " value " << ll->val << std::endl;
    while(ll->next!=ll){
        ll = ll->next;
        std::cout << "col " << i << " row " << ll->row << " value " << ll->val << std::endl;
    }
}

DegreePQ* PriorityPreconditioner::getDegreePQ(std::vector<double> degs){
    // std::cout << "get the PQ" << std::endl;
    double n = degs.size();
    std::vector<DegreePQElement*> elems;
    for(int i = 0; i < n; i++){
        elems.push_back(nullptr);
    }
    std::vector<double> lists;
    for(int i = 0; i < 2*n + 1; i++){
        lists.push_back(-1);
    }
    double minlist = 0;

    for(int i = 0; i < n; i++){
        double key = degs.at(i);
        double head = lists.at(key);

        if(head >= 0){
            DegreePQElement* elem_i = new DegreePQElement();
            elem_i->prev = -1;
            elem_i->next = head;
            elem_i->key = key;
            elems.at(i) = elem_i;

            // DegreePQElement* elem_head = new DegreePQElement();
            // elem_head->prev = i;
            // elem_head->next = elems.at(head)->next;
            // elem_head->key = elems.at(head)->key;
            // elems.at(head) = elem_head;
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
    // std::cout << "got the PQ" << std::endl;
    return pq;
}

double PriorityPreconditioner::DegreePQPop(DegreePQ* pq){
    // std::cout << "popping the element" << std::endl;
    if(pq->nitems == 0){
        throw std::invalid_argument("the PQ is empty. Cannot pop an element");
    }
    // std::cout << " minlist before pop = " << pq->minlist << std::endl;
    while (pq->lists.at(pq->minlist) == -1){
        pq->minlist += 1;
    }
    // std::cout << " minlist after pop = " << pq->minlist << std::endl;
    // pq->minlist now represents the min degree of the nodes in the graph
    double i = pq->lists.at(pq->minlist);
    // i is the node with min degree and will be popped
    double next = pq->elems.at(i)->next;
    // std::cout << " next of the popped element = " << next << std::endl;
    // update the index of the node with min-degree in pq->lists
    pq->lists.at(pq->minlist) = next;

    delete pq->elems.at(i);
    pq->elems.at(i) = nullptr;
    if(next > -1){
        // DegreePQElement* elem_next = new DegreePQElement();
        // elem_next->prev = -1;
        // elem_next->next = pq->elems.at(next)->next;
        // elem_next->key = pq->elems.at(next)->key;
        // pq->elems.at(next) = elem_next;
        pq->elems.at(next)->prev = -1;
    }

    pq->nitems -= 1;
    // return the index of the node in elems to be popped out.
    // std::cout << "popped the element " << i << std::endl;
    return i;
}

void PriorityPreconditioner::DegreePQMove(DegreePQ* pq, int i, double newkey, int oldlist, int newlist){
    // std::cout << "move in the PQ" << std::endl;
    // std::cout << "  i = " << i << " newkey = " << newkey << " oldlist = " << oldlist << " newlist = " << newlist << std::endl;
    double prev = pq->elems.at(i)->prev;
    double next = pq->elems.at(i)->next;
    // std::cout << "prev = " << prev << " next = " << next << std::endl;
    // remove i from the oldlist
    if(next > -1){
        // DegreePQElement* elem_next = new DegreePQElement();
        // elem_next->prev = prev;
        // elem_next->next = pq->elems.at(next)->next;
        // elem_next->key = pq->elems.at(next)->key;
        // pq->elems.at(next) = elem_next;
        pq->elems.at(next)->prev = prev;
    }
    if(prev > -1){
        // DegreePQElement* elem_prev = new DegreePQElement();
        // elem_prev->prev = pq->elems.at(prev)->prev;
        // elem_prev->next = next;
        // elem_prev->key = pq->elems.at(prev)->key;
        // pq->elems.at(prev) = elem_prev;
        pq->elems.at(prev)->next = next;
    } else{
        pq->lists.at(oldlist) = next;
    }

    // std::cout<< "insert " << i << " into newlist" << std::endl;
    double head = pq->lists.at(newlist);
    // std::cout << " head = " << head << std::endl;
    if(head > -1){
        // DegreePQElement* elem_head = new DegreePQElement();
        // elem_head->prev = i;
        // elem_head->next = pq->elems.at(head)->next;
        // elem_head->key = pq->elems.at(head)->key;
        // pq->elems.at(head) = elem_head;
        pq->elems.at(head)->prev = i;
    }
    pq->lists.at(newlist) = i;

    // DegreePQElement* elem_i = new DegreePQElement();
    // elem_i->prev = -1;
    // elem_i->next = head;
    // elem_i->key = newkey;
    // pq->elems.at(i) = elem_i;
    pq->elems.at(i)->prev = -1;
    pq->elems.at(i)->next = head;
    pq->elems.at(i)->key = newkey;
    // std::cout << "moved in the PQ" << std::endl;
    return;
}

void PriorityPreconditioner::DegreePQDec(DegreePQ* pq, int i){
    // std::cout << "dec in the PQ " << i << std::endl;
    double n = pq->n;
    double deg_i = pq->elems.at(i)->key;
    if(deg_i == 1){
        return;
    }
    int oldlist = deg_i <= n ? deg_i : n + int(deg_i/n);
    int newlist = deg_i-1 <= n ? deg_i-1 : n + int((deg_i-1)/n);
    // std::cout << " deg_i = " << deg_i << std::endl;
    if(oldlist != newlist){
        DegreePQMove(pq, i, deg_i-1, oldlist, newlist);
        if(newlist < pq->minlist){
            pq->minlist = newlist;
        }
    }else{
        // DegreePQElement* elem_i = new DegreePQElement();
        // elem_i->prev = pq->elems.at(i)->prev;
        // elem_i->next = pq->elems.at(i)->next;
        // elem_i->key = pq->elems.at(i)->key - 1;
        // pq->elems.at(i) = elem_i;
        pq->elems.at(i)->key -= 1;
    }
    // std::cout << "deced in the PQ" << std::endl;
    return;
}

void PriorityPreconditioner::DegreePQInc(DegreePQ* pq, int i){
    // std::cout << "inc in the PQ" << std::endl;
    double n = pq->n;
    double deg_i = pq->elems.at(i)->key;
    int oldlist = deg_i <= n ? deg_i : n + int(deg_i/n);
    int newlist = deg_i+1 <= n ? deg_i+1 : n + int((deg_i+1)/n);
    // std::cout << " deg_i = " << deg_i << std::endl;
    if(oldlist != newlist){
        DegreePQMove(pq, i, deg_i+1, oldlist, newlist);
    }else{
        // DegreePQElement* elem_i = new DegreePQElement();
        // elem_i->prev = pq->elems.at(i)->prev;
        // elem_i->next = pq->elems.at(i)->next;
        // elem_i->key = pq->elems.at(i)->key + 1;
        // pq->elems.at(i) = elem_i;
        pq->elems.at(i)->key += 1;
    }
    // std::cout << "inced in the PQ" << std::endl;
    return;
}

double PriorityPreconditioner::getColumnLength(PriorityMatrix* pmat, int i, std::vector<PriorityElement*>* colspace){
    // std::cout << "get length for col " << i << std::endl;
    PriorityElement* ll = pmat->cols[i];
    double len = 0;
    while(ll->next != ll){
        // std::cout << "loop in gcl" << std::endl;
        if(ll->val > 0){
            len += 1;
            if(len > colspace->size()){
                colspace->push_back(ll);
            } else{
                colspace->at(len-1) = ll;
            }
        }
        // std::cout << " row : " << ll->row << std::endl;
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
    // std::cout << "got col length = " << len << std::endl;
    return len;
}

double PriorityPreconditioner::compressColumn(std::vector<PriorityElement*>* colspace, double len, DegreePQ* pq){
    // std::cout << "Compressing col of len = " << len << std::endl;

    std::sort(colspace->begin(), colspace->begin()+len,
         [](PriorityElement* j, PriorityElement* k) {return j->row < k->row; });

    double ptr = PTR_RESET;
    double currow = -1;

    for(int i = 0; i < len; i++){
        if(colspace->at(i)->row != currow){
            // std::cout << "currrow != colspace[i].row " << currow << " " << colspace->at(i)->row << std::endl;

            currow = colspace->at(i)->row;
            ptr += 1;
            colspace->at(ptr) = colspace->at(i);
        }
        else{
            // std::cout << "currrow == colspace[i]->row " << currow << std::endl;
            colspace->at(ptr)->val += colspace->at(i)->val;
            colspace->at(i)->reverse->val = 0;

            DegreePQDec(pq, currow);
        }
    }

    std::sort(colspace->begin(), colspace->begin()+ptr+1,
         [](PriorityElement* j, PriorityElement* k) {return j->val < k->val; });
    //  std::cout << "Compressed col to len = " << ptr+1 << std::endl;
    return ptr+1;
}

double PriorityPreconditioner::compressColumnSC(std::vector<PriorityElement*>* colspace, double len, DegreePQ* pq){
    // std::cout << "Compressing col of len = " << len << std::endl;

    std::sort(colspace->begin(), colspace->begin()+len,
         [](PriorityElement* j, PriorityElement* k) {return j->row < k->row; });

    double ptr = PTR_RESET;
    double currow = -1;

    for(int i = 0; i < len; i++){
        if(colspace->at(i)->row != currow){
            // std::cout << "currrow != colspace[i].row " << currow << " " << colspace->at(i)->row << std::endl;

            currow = colspace->at(i)->row;
            ptr += 1;
            colspace->at(ptr) = colspace->at(i);
        }
        else{
            // std::cout << "currrow == colspace[i]->row " << currow << std::endl;
            colspace->at(ptr)->val += colspace->at(i)->val;
            // colspace->at(i)->reverse->val = 0;

            // DegreePQDec(pq, currow);
        }
    }

    std::sort(colspace->begin(), colspace->begin()+ptr+1,
         [](PriorityElement* j, PriorityElement* k) {return j->val < k->val; });
    //  std::cout << "Compressed col to len = " << ptr+1 << std::endl;
    return ptr+1;
}


LDLi* PriorityPreconditioner::getLDLi(){
    PriorityMatrix* a = getPriorityMatrix();
    double n = a->n;
    LDLi* ldli = new LDLi();
    ldli->col = std::vector<double>();
    ldli->colptr = std::vector<double>();
    ldli->rowval = std::vector<double>();
    ldli->fval = std::vector<double>();

    double ldli_row_ptr = 0;
    std::vector<double> d(n, 0.0);

    DegreePQ* pq = getDegreePQ(a->degs);

    double it = 1;
    std::vector<PriorityElement*>* colspace = new std::vector<PriorityElement*>();
    std::mt19937_64 rand_generator;
    std::uniform_real_distribution<double> u_distribution(0, 1);
    while(it < n){
        // std::cout << "looop " << it << std::endl;
        double i = DegreePQPop(pq);

        ldli->col.push_back(i);
        ldli->colptr.push_back(ldli_row_ptr);

        it += 1;

        double len = getColumnLength(a, i, colspace);
        len = compressColumn(colspace, len, pq);
        double csum = 0;
        std::vector<double> cumspace;
        std::vector<double> vals;
        for(int ii = 0; ii < len ; ii++){
            vals.push_back(colspace->at(ii)->val);
            csum += colspace->at(ii)->val;
            cumspace.push_back(csum);
        }
        double wdeg = csum;
        double colScale = 1;

        for(int joffset = 0; joffset < len-1; joffset++){
            PriorityElement* ll = colspace->at(joffset);
            double w = vals.at(joffset) * colScale;
            double j = ll->row;
            PriorityElement* revj = ll->reverse;
            double f = double(w)/wdeg;
            vals.at(joffset) = 0;

            double u_r = u_distribution(rand_generator);
            double r = u_r*(csum  - cumspace[joffset]) + cumspace[joffset];
            double koff = len-1;
            for(int k_i = 0; k_i < len; k_i++){
                if(cumspace[k_i]>r){
                    koff = k_i;
                    break;
                }
            }
            double k = colspace->at(koff)->row;

            // A new edge is being added from node j to node k.
            // However, since the edge node i to node j is being removed
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

            ldli->rowval.push_back(j);
            ldli->fval.push_back(f);
            ldli_row_ptr += 1;
        }

        if(len > 0){
            // std::cout << "len = " << len << "vals len = " << vals.size() << " colspace len = " << colspace->size() << std::endl;
            PriorityElement* ll = colspace->at(len-1);
            double w = vals.at(len-1)*colScale;
            double j = ll->row;
            PriorityElement* revj = ll->reverse;
            if(it < n){
                // This indicates that the edge from node i to node j
                // has been removed
                DegreePQDec(pq, j);
            }
            ll->val = 0;
            revj->val = 0;

            ldli->rowval.push_back(j);
            ldli->fval.push_back(1.0);
            ldli_row_ptr += 1;

            d[i] = w;
        }
    }
    ldli->colptr.push_back(ldli_row_ptr);
    ldli->d = d;

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

    return ldli;
}


Eigen::MatrixXd PriorityPreconditioner::getSchurComplement(int t){
    PriorityMatrix* a = getPriorityMatrix();
    double n = a->n;

    DegreePQ* pq = getDegreePQ(a->degs);

    double it = 1;
    std::vector<PriorityElement*>* colspace = new std::vector<PriorityElement*>();
    std::mt19937_64 rand_generator;
    std::uniform_real_distribution<double> u_distribution(0, 1);
    while(it <= t && it < n){
        // std::cout << "looop " << it << std::endl;
        double i = DegreePQPop(pq);
        // std::cout << "Eliminated node " << i << std::endl;

        it += 1;

        double len = getColumnLength(a, i, colspace);
        len = compressColumn(colspace, len, pq);
        double csum = 0;
        std::vector<double> cumspace;
        std::vector<double> vals;
        for(int ii = 0; ii < len ; ii++){
            vals.push_back(colspace->at(ii)->val);
            csum += colspace->at(ii)->val;
            cumspace.push_back(csum);
        }
        double wdeg = csum;
        double colScale = 1;

        for(int joffset = 0; joffset < len-1; joffset++){
            PriorityElement* ll = colspace->at(joffset);
            double w = vals.at(joffset) * colScale;
            double j = ll->row;
            PriorityElement* revj = ll->reverse;
            double f = double(w)/wdeg;
            vals.at(joffset) = 0;

            double u_r = u_distribution(rand_generator);
            double r = u_r*(csum  - cumspace[joffset]) + cumspace[joffset];
            double koff = len-1;
            for(int k_i = 0; k_i < len; k_i++){
                if(cumspace[k_i]>r){
                    koff = k_i;
                    break;
                }
            }
            double k = colspace->at(koff)->row;

            // A new edge is being added from node j to node k.
            // However, since the edge node i to node j is being removed
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
            // std::cout << "len = " << len << "vals len = " << vals.size() << " colspace len = " << colspace->size() << std::endl;
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

    // std::cout << "REMAINING ELEMENTS: " << pq->nitems << std::endl;
    // std::cout << "PQ minlist: " << pq->minlist << std::endl;
    int edge_info_counter = 0;
    while (pq->nitems > 0){
        double i = DegreePQPop(pq);
        double len = getColumnLength(a, i, colspace);
        // std::cout << "Column: " << i << " length: " << len << std::endl;
        len = compressColumnSC(colspace, len, pq);
        // std::cout << "Column: " << i << " compressed length: " << len << std::endl;
        for(int ii = 0; ii < len ; ii++){
            double val = colspace->at(ii)->val;
            double row = colspace->at(ii)->row;
            Eigen::Vector3d temp(row, i, val);
            edge_info_vector.push_back(temp);
            // edge_info.row(edge_info_counter) << row, i, val;
            edge_info_counter += 1;
        }
    }
    edge_info.resize(edge_info_counter, 3);
    for(int z = 0; z < edge_info_counter; z++){
        // std::cout << "ROW: " << edge_info_vector[z] << std::endl;
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

// RandomPreconditioner


RandomPreconditioner::RandomPreconditioner(Eigen::SparseMatrix<double>* A, std::string o_n){
    _A = A;
    _o_n_str = o_n;
}

std::vector<double> RandomPreconditioner::getFlipIndices(Eigen::SparseMatrix<double>* M){
    Eigen::SparseMatrix<double> F = Eigen::SparseMatrix<double>(M->rows(), M->cols());
    std::vector<Eigen::Triplet<double> > triplets;

    int n = M->rows();
    double counter = 0;
    for(int coln = 0; coln < n; coln++){
        int start_idx = M->outerIndexPtr()[coln];
        int end_idx = M->outerIndexPtr()[coln+1];
        for(int ind = start_idx ; ind < end_idx; ind++){
            int rown  = M->innerIndexPtr()[ind];
            triplets.push_back(Eigen::Triplet<double>(rown, coln, counter));
            counter += 1;
        }
    }
    F.setFromTriplets(triplets.begin(), triplets.end());
    // std::cout << "F matrix = \n" << F << std::endl;
    Eigen::SparseMatrix<double> F_t = F.transpose();
    // std::cout << "F_t matrix = \n" << F_t << std::endl;
    std::vector<double> flipped_indices;
    for(int coln = 0; coln < n; coln++){
        int start_idx = F_t.outerIndexPtr()[coln];
        int end_idx = F_t.outerIndexPtr()[coln+1];
        for(int ind = start_idx ; ind < end_idx; ind++){
            flipped_indices.push_back(F_t.valuePtr()[ind]);
        }
    }
    return flipped_indices;
}

void RandomPreconditioner::printFlipIndices(std::vector<double> fi){
    std::cout << "flip indices = " << std::endl;
    for(int i = 0; i < fi.size(); i++){
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
    // std::cout << "Getting PriorityMatrix " << std::endl;
    int n = _A->rows();
    std::vector<PriorityElement*> cols;
    std::vector<PriorityElement*> llelems;
    std::vector<double> degs;

    // std::cout << "flipping " << std::endl;
    std::vector<double> flips = getFlipIndices(_A);
    // printFlipIndices(flips);
    // std::cout << "flipped " << std::endl;

    for(int i = 0; i < n; i++){
        int start_idx = _A->outerIndexPtr()[i];
        int end_idx = _A->outerIndexPtr()[i+1];
        double deg = end_idx - start_idx;
        degs.push_back(deg);
        if(deg == 0){
            // std::cout << " deg[" << i << "] = " <<  end_idx - start_idx << std::endl;
            PriorityElement* pe = new PriorityElement();
            cols.push_back(pe);
            continue;
        }
        // std::cout << " deg[" << i << "] = " <<  end_idx - start_idx << std::endl;

        double j = _A->innerIndexPtr()[start_idx];
        double v = _A->valuePtr()[start_idx];
        PriorityElement* pe = new PriorityElement(j, v);
        llelems.push_back(pe);
        PriorityElement* next = pe;

        for(int ind = start_idx + 1; ind < end_idx; ind++){
            j = _A->innerIndexPtr()[ind];
            v = _A->valuePtr()[ind];
            PriorityElement* pe = new PriorityElement(j, v, next);
            llelems.push_back(pe);
            next = pe;
        }
        cols.push_back(next);
    }

    for(int i = 0; i < n; i++){
        int start_idx = _A->outerIndexPtr()[i];
        int end_idx = _A->outerIndexPtr()[i+1];
        for(int ind = start_idx; ind < end_idx; ind++){
            llelems.at(ind)->reverse = llelems.at(flips.at(ind));
        }
    }

    PriorityMatrix* pmat = new PriorityMatrix();
    pmat->n = n;
    pmat->degs = degs;
    pmat->cols = cols;
    pmat->lles = llelems;
    // std::cout << "Got LLMAtp " << "degs size = " << degs.size() << std::endl;
    return pmat;
}

void RandomPreconditioner::printColumn(PriorityMatrix* pmat, int i){
    PriorityElement* ll = pmat->cols.at(i);
    std::cout << "col " << i << " row " << ll->row << " value " << ll->val << std::endl;
    while(ll->next!=ll){
        ll = ll->next;
        std::cout << "col " << i << " row " << ll->row << " value " << ll->val << std::endl;
    }
}

RandomPQ* RandomPreconditioner::getRandomPQ(std::vector<double> degs){
    // std::cout << "get the PQ" << std::endl;
    double n = degs.size();
    std::vector<double> node_id;
    for(int i = 0; i < n; i++){
        node_id.push_back(i);
    }
    std::random_shuffle ( node_id.begin(), node_id.end() );
    RandomPQ* pq = new RandomPQ();
    pq->node_id = node_id;
    pq->nitems = n;
    return pq;
}

double RandomPreconditioner::RandomPQPop(RandomPQ* pq){
    // std::cout << "popping the element" << std::endl;
    if(pq->nitems == 0){
        throw std::invalid_argument("the PQ is empty. Cannot pop an element");
    }
    // std::cout << " minlist before pop = " << pq->minlist << std::endl;
    double i = pq->node_id[pq->nitems - 1];
    pq->nitems -= 1;
    // return the index of the node in elems to be popped out.
    // std::cout << "popped the element " << i << std::endl;
    return i;
}


double RandomPreconditioner::getColumnLength(PriorityMatrix* pmat, int i, std::vector<PriorityElement*>* colspace){
    // std::cout << "get length for col " << i << std::endl;
    PriorityElement* ll = pmat->cols[i];
    double len = 0;
    while(ll->next != ll){
        // std::cout << "loop in gcl" << std::endl;
        if(ll->val > 0){
            len += 1;
            if(len > colspace->size()){
                colspace->push_back(ll);
            } else{
                colspace->at(len-1) = ll;
            }
        }
        // std::cout << " row : " << ll->row << std::endl;
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
    // std::cout << "got col length = " << len << std::endl;
    return len;
}

double RandomPreconditioner::compressColumn(std::vector<PriorityElement*>* colspace, double len){
    // std::cout << "Compressing col of len = " << len << std::endl;

    std::sort(colspace->begin(), colspace->begin()+len,
         [](PriorityElement* j, PriorityElement* k) {return j->row < k->row; });

    double ptr = PTR_RESET;
    double currow = -1;

    for(int i = 0; i < len; i++){
        if(colspace->at(i)->row != currow){
            // std::cout << "currrow != colspace[i].row " << currow << " " << colspace->at(i)->row << std::endl;

            currow = colspace->at(i)->row;
            ptr += 1;
            colspace->at(ptr) = colspace->at(i);
        }
        else{
            // std::cout << "currrow == colspace[i]->row " << currow << std::endl;
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
        std::random_shuffle ( colspace->begin(), colspace->begin()+ptr+1 );
    }
    //  std::cout << "Compressed col to len = " << ptr+1 << std::endl;
    return ptr+1;
}

double RandomPreconditioner::compressColumnSC(std::vector<PriorityElement*>* colspace, double len){
    // std::cout << "Compressing col of len = " << len << std::endl;

    std::sort(colspace->begin(), colspace->begin()+len,
         [](PriorityElement* j, PriorityElement* k) {return j->row < k->row; });

    double ptr = PTR_RESET;
    double currow = -1;

    for(int i = 0; i < len; i++){
        if(colspace->at(i)->row != currow){
            // std::cout << "currrow != colspace[i].row " << currow << " " << colspace->at(i)->row << std::endl;

            currow = colspace->at(i)->row;
            ptr += 1;
            colspace->at(ptr) = colspace->at(i);
        }
        else{
            // std::cout << "currrow == colspace[i]->row " << currow << std::endl;
            colspace->at(ptr)->val += colspace->at(i)->val;
            // colspace->at(i)->reverse->val = 0;
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
        std::random_shuffle ( colspace->begin(), colspace->begin()+ptr+1 );
    }
    //  std::cout << "Compressed col to len = " << ptr+1 << std::endl;
    return ptr+1;
}


LDLi* RandomPreconditioner::getLDLi(){
    PriorityMatrix* a = getPriorityMatrix();
    double n = a->n;
    LDLi* ldli = new LDLi();
    ldli->col = std::vector<double>();
    ldli->colptr = std::vector<double>();
    ldli->rowval = std::vector<double>();
    ldli->fval = std::vector<double>();

    double ldli_row_ptr = 0;
    std::vector<double> d(n, 0.0);

    RandomPQ* pq = getRandomPQ(a->degs);

    double it = 1;
    std::vector<PriorityElement*>* colspace = new std::vector<PriorityElement*>();
    std::mt19937_64 rand_generator;
    std::uniform_real_distribution<double> u_distribution(0, 1);
    while(it < n){
        // std::cout << "looop " << it << std::endl;
        double i = RandomPQPop(pq);

        ldli->col.push_back(i);
        ldli->colptr.push_back(ldli_row_ptr);

        it += 1;

        double len = getColumnLength(a, i, colspace);
        len = compressColumn(colspace, len);
        double csum = 0;
        std::vector<double> cumspace;
        std::vector<double> vals;
        for(int ii = 0; ii < len ; ii++){
            vals.push_back(colspace->at(ii)->val);
            csum += colspace->at(ii)->val;
            cumspace.push_back(csum);
        }
        double wdeg = csum;
        double colScale = 1;

        for(int joffset = 0; joffset < len-1; joffset++){
            PriorityElement* ll = colspace->at(joffset);
            double w = vals.at(joffset) * colScale;
            double j = ll->row;
            PriorityElement* revj = ll->reverse;
            double f = double(w)/wdeg;
            vals.at(joffset) = 0;

            double u_r = u_distribution(rand_generator);
            double r = u_r*(csum  - cumspace[joffset]) + cumspace[joffset];
            double koff = len-1;
            for(int k_i = 0; k_i < len; k_i++){
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

            ldli->rowval.push_back(j);
            ldli->fval.push_back(f);
            ldli_row_ptr += 1;
        }

        if(len > 0){
            // std::cout << "len = " << len << "vals len = " << vals.size() << " colspace len = " << colspace->size() << std::endl;
            PriorityElement* ll = colspace->at(len-1);
            double w = vals.at(len-1)*colScale;
            double j = ll->row;
            PriorityElement* revj = ll->reverse;

            ll->val = 0;
            revj->val = 0;

            ldli->rowval.push_back(j);
            ldli->fval.push_back(1.0);
            ldli_row_ptr += 1;

            d[i] = w;
        }
    }
    ldli->colptr.push_back(ldli_row_ptr);
    ldli->d = d;

    // delete colspace
    delete colspace;
    colspace = nullptr;

    // // delete left-over pointers to pq elements
    // for(auto pq_element: pq->elems){
    //     delete pq_element;
    //     pq_element = nullptr;
    // }
    delete pq;
    pq = nullptr;

    // delete the pointers in priority matrix
    for(auto ll_element: a->lles){
        delete ll_element;
        ll_element = nullptr;
    }
    delete a;
    a = nullptr;

    return ldli;
}


// NOTE: The stochasticity of this strategy can lead to non-deterministic
// sizes in schur complements. For example, if a node with degree 1 is eliminated
// there will be no nodes left for a clique to be formed. This might lead to
// a slightly lower number of nodes than expected. Such issues can be addressed
// using degree based elimination orders.
Eigen::MatrixXd RandomPreconditioner::getSchurComplement(int t){
    PriorityMatrix* a = getPriorityMatrix();
    double n = a->n;

    RandomPQ* pq = getRandomPQ(a->degs);

    double it = 1;
    std::vector<PriorityElement*>* colspace = new std::vector<PriorityElement*>();
    std::mt19937_64 rand_generator;
    std::uniform_real_distribution<double> u_distribution(0, 1);
    while(it <= t && it < n){
        // std::cout << "looop " << it << std::endl;
        double i = RandomPQPop(pq);
        // std::cout << "Eliminated node " << i << std::endl;

        double len = getColumnLength(a, i, colspace);
        len = compressColumn(colspace, len);
        double csum = 0;
        std::vector<double> cumspace;
        std::vector<double> vals;
        for(int ii = 0; ii < len ; ii++){
            vals.push_back(colspace->at(ii)->val);
            csum += colspace->at(ii)->val;
            cumspace.push_back(csum);
        }
        double wdeg = csum;
        double colScale = 1;

        // std::cout << "Length of joffsets: " << len << std::endl;
        for(int joffset = 0; joffset < len-1; joffset++){
            PriorityElement* ll = colspace->at(joffset);
            double w = vals.at(joffset) * colScale;
            double j = ll->row;
            PriorityElement* revj = ll->reverse;
            double f = double(w)/wdeg;
            vals.at(joffset) = 0;

            double u_r = u_distribution(rand_generator);
            double r = u_r*(csum  - cumspace[joffset]) + cumspace[joffset];
            double koff = len-1;
            for(int k_i = 0; k_i < len; k_i++){
                if(cumspace[k_i]>r){
                    koff = k_i;
                    break;
                }
            }
            double k = colspace->at(koff)->row;

            double newEdgeVal = f*(1-f)*wdeg;
            // std::cout << "NEW EDGE: " << k << " " << j << " " << newEdgeVal << std::endl;

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
            // std::cout << "--------------------------------------------------" << std::endl;
            // std::cout << "a->cols.at(k) val" <<  ll->val << " " << a->cols.at(k)->val << std::endl;
            // PriorityElement* temp = a->cols[k];
            // while(temp->next != temp){
            //     std::cout << "ROW: " << temp->row << " VAL: " << temp->val << std::endl;
            //     temp = temp->next;
            // }
            // std::cout << "ROW: " << temp->row << " VAL: " << temp->val << std::endl;
            // std::cout << "--------------------------------------------------" << std::endl;

            colScale = colScale*(1-f);
            wdeg = wdeg*(1-f)*(1-f);
        }

        if(len > 0){
            // std::cout << "len = " << len << "vals len = " << vals.size() << " colspace len = " << colspace->size() << std::endl;
            PriorityElement* ll = colspace->at(len-1);
            // double j = ll->row;
            PriorityElement* revj = ll->reverse;
            ll->val = 0;
            revj->val = 0;

        }

        it += 1;
    }

    // std::cout << "------------------------------- PREPARING SC -------------------------------" << std::endl;
    std::vector<Eigen::Vector3d> edge_info_vector;
    Eigen::MatrixXd edge_info;

    // std::cout << "REMAINING ELEMENTS: " << pq->nitems << std::endl;
    int edge_info_counter = 0;
    while (pq->nitems > 0){
        double i = RandomPQPop(pq);
        // std::cout << "Element: " << i << std::endl;
        double len = getColumnLength(a, i, colspace);
        // std::cout << "Column: " << i << " length: " << len << std::endl;
        // std::cout << "--------------------------------------------------" << std::endl;
        // PriorityElement* temp = a->cols[i];
        // while(temp->next != temp){
        //     std::cout << "ROW: " << temp->row << " VAL: " << temp->val << std::endl;
        //     temp = temp->next;
        // }
        // std::cout << "ROW: " << temp->row << " VAL: " << temp->val << std::endl;
        // std::cout << "--------------------------------------------------" << std::endl;
        len = compressColumnSC(colspace, len);
        // std::cout << "Column: " << i << " compressed length: " << len << std::endl;
        for(int ii = 0; ii < len ; ii++){
            double val = colspace->at(ii)->val;
            double row = colspace->at(ii)->row;
            Eigen::Vector3d temp(row, i, val);
            edge_info_vector.push_back(temp);
            // edge_info.row(edge_info_counter) << row, i, val;
            edge_info_counter += 1;
        }
    }
    edge_info.resize(edge_info_counter, 3);
    for(int z = 0; z < edge_info_counter; z++){
        // std::cout << "ROW: " << edge_info_vector[z] << std::endl;
        edge_info.row(z) = edge_info_vector[z];
    }

    // delete colspace
    delete colspace;
    colspace = nullptr;

    // // delete left-over pointers to pq elements
    // for(auto pq_element: pq->elems){
    //     delete pq_element;
    //     pq_element = nullptr;
    // }
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


// CoarseningPreconditioner

CoarseningPreconditioner::CoarseningPreconditioner(Eigen::SparseMatrix<double>* A):
    PriorityPreconditioner(A){
    _A = A;
}

LDLi* CoarseningPreconditioner::getLDLi(){
    PriorityMatrix* a = getPriorityMatrix();
    double n = a->n;
    LDLi* ldli = new LDLi();
    ldli->col = std::vector<double>();
    ldli->colptr = std::vector<double>();
    ldli->rowval = std::vector<double>();
    ldli->fval = std::vector<double>();

    double ldli_row_ptr = 0;
    std::vector<double> d(n, 0.0);

    DegreePQ* pq = getDegreePQ(a->degs);

    double it = 1;
    std::vector<PriorityElement*>* colspace = new std::vector<PriorityElement*>();
    std::mt19937_64 rand_generator;
    std::uniform_real_distribution<double> u_distribution(0, 1);
    while(it < n){
        double i = DegreePQPop(pq);

        ldli->col.push_back(i);
        ldli->colptr.push_back(ldli_row_ptr);

        it += 1;

        double len = getColumnLength(a, i, colspace);
        len = compressColumn(colspace, len, pq);
        // continue to the next node if there are no edges for this node
        if(len < 1) continue;

        double csum = 0;
        std::vector<double> cumspace;
        std::vector<double> vals;
        for(int ii = 0; ii < len ; ii++){
            vals.push_back(colspace->at(ii)->val);
            csum += colspace->at(ii)->val;
            cumspace.push_back(csum);
        }
        double wdeg = csum;

        // choose the neighbour with probability proportional
        // to its weight
        double u_r = u_distribution(rand_generator);
        double r = u_r*(csum);
        double koff = len-1;
        for(int k_i = 0; k_i < len; k_i++){
            if(cumspace[k_i]>r){
                koff = k_i;
                break;
            }
        }
        double k = colspace->at(koff)->row;
        double w_k = vals.at(koff);
        // std::cout << " it = " << it << " Chose random vertex: " << k << " offset " << koff << std::endl;
        PriorityElement* k_pe = colspace->at(koff);
        // collapse the current node onto the selected neighbour
        k_pe->val = 0;
        k_pe->reverse->val = 0;
        DegreePQDec(pq, k);
        // remove edges from node i to it's neighbours and
        // add random edges among it's neighbours from the selected
        // neighbour k
        for(int joffset = 0; joffset < len; joffset++){
            if(joffset == koff) continue;

            PriorityElement* ll = colspace->at(joffset);
            // double w = vals.at(joffset) * colScale;
            double w = vals.at(joffset);
            double j = ll->row;
            // std::cout << " j  = " << j << std::endl;
            PriorityElement* revj = ll->reverse;

            double f = double(w)/wdeg;
            vals.at(joffset) = 0;

            DegreePQInc(pq, k);

            // double newEdgeVal = f*(1-f)*wdeg;
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

            ldli->rowval.push_back(j);
            ldli->fval.push_back(f);
            ldli_row_ptr += 1;
        }

        if(len > 0){
            // std::cout << "len = " << len << "vals len = " << vals.size() << " colspace len = " << colspace->size() << std::endl;
            // double w = vals.at(len-1)*colScale;
            double w = vals.at(len-1);
            d[i] = w;
        }
    }
    ldli->colptr.push_back(ldli_row_ptr);
    ldli->d = d;

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
    return ldli;
}


Eigen::MatrixXd CoarseningPreconditioner::getSchurComplement(int t){

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
        for(int ii = 0; ii < len ; ii++){
            vals.push_back(colspace->at(ii)->val);
            csum += colspace->at(ii)->val;
            cumspace.push_back(csum);
        }
        // double wdeg = csum;

        // choose the neighbour with probability proportional
        // to its weight
        double u_r = u_distribution(rand_generator);
        double r = u_r*(csum);
        double koff = len-1;
        for(int k_i = 0; k_i < len; k_i++){
            if(cumspace[k_i]>r){
                koff = k_i;
                break;
            }
        }
        double k = colspace->at(koff)->row;
        double w_k = vals.at(koff);
        // std::cout << " it = " << it << " Chose random vertex: " << k << " offset " << koff << std::endl;
        PriorityElement* k_pe = colspace->at(koff);
        // collapse the current node onto the selected neighbour
        k_pe->val = 0;
        k_pe->reverse->val = 0;
        DegreePQDec(pq, k);
        // remove edges from node i to it's neighbours and
        // add random edges among it's neighbours from the selected
        // neighbour k
        for(int joffset = 0; joffset < len; joffset++){
            if(joffset == koff) continue;

            PriorityElement* ll = colspace->at(joffset);
            // double w = vals.at(joffset) * colScale;
            double w = vals.at(joffset);
            double j = ll->row;
            // std::cout << " j  = " << j << std::endl;
            PriorityElement* revj = ll->reverse;

            // double f = double(w)/wdeg;
            vals.at(joffset) = 0;

            DegreePQInc(pq, k);

            // double newEdgeVal = f*(1-f)*wdeg;
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
    // std::cout << "REMAINING ELEMENTS: " << pq->nitems << std::endl;
    int edge_info_counter = 0;
    while (pq->nitems > 0){
        double i = DegreePQPop(pq);
        double len = getColumnLength(a, i, colspace);
        // std::cout << "Column: " << i << " length: " << len << std::endl;
        len = compressColumnSC(colspace, len, pq);
        // std::cout << "Column: " << i << " compressed length: " << len << std::endl;
        for(int ii = 0; ii < len ; ii++){
            double val = colspace->at(ii)->val;
            double row = colspace->at(ii)->row;
            Eigen::Vector3d temp(row, i, val);
            edge_info_vector.push_back(temp);
            // edge_info.row(edge_info_counter) << row, i, val;
            edge_info_counter += 1;
        }
    }
    edge_info.resize(edge_info_counter, 3);
    for(int z = 0; z < edge_info_counter; z++){
        // std::cout << "ROW: " << edge_info_vector[z] << std::endl;
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