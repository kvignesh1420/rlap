#include <iostream>
#include <vector>
#include <functional>
#include <random>
#include <stdexcept>
#include "third_party/eigen3/Eigen/SparseCore"
#include "preconditioner.h"
#include "types.h"

#define PTR_RESET -1

OrderedPreconditioner::OrderedPreconditioner(Eigen::SparseMatrix<float>* A){
    _A = A;
    _ordmat = getOrderedMatrix();
}


OrderedMatrix* OrderedPreconditioner::getOrderedMatrix(){
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

float OrderedPreconditioner::getColumnLength(OrderedMatrix* ordmat, int i, std::vector<ColumnElement>* colspace){
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

void OrderedPreconditioner::printColumn(OrderedMatrix* ordmat, int i){
    float ptr = ordmat->cols[i];
    std::cout << "printing col " << i << std::endl;
    while(ptr != PTR_RESET){
        OrderedElement* oe = ordmat->elements[ptr];
        std::cout << "col " << i << " row " << oe->row << " value : " << oe->val << std::endl;
        ptr = oe->next;
    }
}

void OrderedPreconditioner::printColspace(std::vector<ColumnElement>* colspace){
    std::cout << "COLSPACE = \n" << std::endl;
    for(int i = 0; i < colspace->size(); i++){
        ColumnElement item = colspace->at(i);
        std::cout << " row " << item.row << " ptr " << item.ptr <<  " value : " << item.val << std::endl;
    }
}

void OrderedPreconditioner::printMatrixDetails(OrderedMatrix* ordmat){
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

float OrderedPreconditioner::compressColumn(std::vector<ColumnElement>* colspace, float len){
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

LDLi* OrderedPreconditioner::getLDLi(){
    float n = _ordmat->n;
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

        float len = getColumnLength(_ordmat, i, colspace);
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
            ColumnElement ce = colspace->at(joffset);
            float w = ce.val*colScale;
            float j = ce.row;
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
                float jhead = _ordmat->cols[j];

                OrderedElement* ord = new OrderedElement();
                ord->row = k;
                ord->next = jhead;
                ord->val = newEdgeVal;
                _ordmat->elements[ce.ptr] = ord;

                _ordmat->cols[j] = ce.ptr;
            } else{
                float khead = _ordmat->cols[k];

                OrderedElement* ord = new OrderedElement();
                ord->row = j;
                ord->next = khead;
                ord->val = newEdgeVal;
                _ordmat->elements[ce.ptr] = ord;

                _ordmat->cols[k] = ce.ptr;
            }

            colScale = colScale*(1-f);
            wdeg = wdeg - 2*w + w*w/wdeg;

            ldli->rowval.push_back(j);
            ldli->fval.push_back(f);
            ldli_row_ptr += 1;
        }

        ColumnElement ce = colspace->at(len-1);
        float w = ce.val*colScale;
        float j = ce.row;
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

// PriorityPreconditioner

PriorityPreconditioner::PriorityPreconditioner(Eigen::SparseMatrix<float>* A){
    _A = A;
    _pmat = getPriorityMatrix();
}

std::vector<float> PriorityPreconditioner::getFlipIndices(Eigen::SparseMatrix<float>* M){
    Eigen::SparseMatrix<float> F = Eigen::SparseMatrix<float>(M->rows(), M->cols());
    std::vector<Eigen::Triplet<float> > triplets;

    int n = M->rows();
    float counter = 0;
    for(int coln = 0; coln < n; coln++){
        int start_idx = M->outerIndexPtr()[coln];
        int end_idx = M->outerIndexPtr()[coln+1];
        for(int ind = start_idx ; ind < end_idx; ind++){
            int rown  = M->innerIndexPtr()[ind];
            triplets.push_back(Eigen::Triplet<float>(rown, coln, counter));
            counter += 1;
        }
    }
    F.setFromTriplets(triplets.begin(), triplets.end());
    // std::cout << "F matrix = \n" << F << std::endl;
    Eigen::SparseMatrix<float> F_t = F.transpose();
    // std::cout << "F_t matrix = \n" << F_t << std::endl;
    std::vector<float> flipped_indices;
    for(int coln = 0; coln < n; coln++){
        int start_idx = F_t.outerIndexPtr()[coln];
        int end_idx = F_t.outerIndexPtr()[coln+1];
        for(int ind = start_idx ; ind < end_idx; ind++){
            flipped_indices.push_back(F_t.valuePtr()[ind]);
        }
    }
    return flipped_indices;
}

void PriorityPreconditioner::printFlipIndices(std::vector<float> fi){
    std::cout << "flip indices = " << std::endl;
    for(int i = 0; i < fi.size(); i++){
        float val = fi.at(i);
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
    std::vector<float> degs;

    // std::cout << "flipping " << std::endl;
    std::vector<float> flips = getFlipIndices(_A);
    // printFlipIndices(flips);
    // std::cout << "flipped " << std::endl;

    for(int i = 0; i < n; i++){
        int start_idx = _A->outerIndexPtr()[i];
        int end_idx = _A->outerIndexPtr()[i+1];
        float deg = end_idx - start_idx;
        degs.push_back(deg);
        if(deg == 0){
            // std::cout << " deg[" << i << "] = " <<  end_idx - start_idx << std::endl;
            PriorityElement* pe = new PriorityElement();
            cols.push_back(pe);
            continue;
        }
        // std::cout << " deg[" << i << "] = " <<  end_idx - start_idx << std::endl;

        float j = _A->innerIndexPtr()[start_idx];
        float v = _A->valuePtr()[start_idx];
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

DegreePQ* PriorityPreconditioner::getDegreePQ(std::vector<float> degs){
    // std::cout << "get the PQ" << std::endl;
    float n = degs.size();
    std::vector<DegreePQElement*> elems;
    for(int i = 0; i < n; i++){
        elems.push_back(nullptr);
    }
    std::vector<float> lists;
    for(int i = 0; i < 2*n + 1; i++){
        lists.push_back(-1);
    }
    float minlist = 0;

    for(int i = 0; i < n; i++){
        float key = degs.at(i);
        float head = lists.at(key);

        if(head >= 0){
            DegreePQElement* elem_i = new DegreePQElement();
            elem_i->prev = -1;
            elem_i->next = head;
            elem_i->key = key;
            elems.at(i) = elem_i;

            DegreePQElement* elem_head = new DegreePQElement();
            elem_head->prev = i;
            elem_head->next = elems.at(head)->next;
            elem_head->key = elems.at(head)->key;
            // TODO: possible memory leak, delete the existing PQ element
            elems.at(head) = elem_head;
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

float PriorityPreconditioner::DegreePQPop(DegreePQ* pq){
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
    float i = pq->lists.at(pq->minlist);
    // i is the node with min degree and will be popped
    float next = pq->elems.at(i)->next;
    // std::cout << " next of the popped element = " << next << std::endl;
    // update the index of the node with min-degree in pq->lists
    pq->lists.at(pq->minlist) = next;

    if(next > -1){
        DegreePQElement* elem_next = new DegreePQElement();
        elem_next->prev = -1;
        elem_next->next = pq->elems.at(next)->next;
        elem_next->key = pq->elems.at(next)->key;
        pq->elems.at(next) = elem_next;
    }

    pq->nitems -= 1;
    // return the index of the node in elems to be popped out.
    // std::cout << "popped the element " << i << std::endl;
    return i;
}

void PriorityPreconditioner::DegreePQMove(DegreePQ* pq, int i, float newkey, int oldlist, int newlist){
    // std::cout << "move in the PQ" << std::endl;
    // std::cout << "  i = " << i << " newkey = " << newkey << " oldlist = " << oldlist << " newlist = " << newlist << std::endl;
    float prev = pq->elems.at(i)->prev;
    float next = pq->elems.at(i)->next;
    // std::cout << "prev = " << prev << " next = " << next << std::endl;
    // remove i from the oldlist
    if(next > -1){
        DegreePQElement* elem_next = new DegreePQElement();
        elem_next->prev = prev;
        elem_next->next = pq->elems.at(next)->next;
        elem_next->key = pq->elems.at(next)->key;
        pq->elems.at(next) = elem_next;
    }
    if(prev > -1){
        DegreePQElement* elem_prev = new DegreePQElement();
        elem_prev->prev = pq->elems.at(prev)->prev;
        elem_prev->next = next;
        elem_prev->key = pq->elems.at(prev)->key;
        pq->elems.at(prev) = elem_prev;
    } else{
        pq->lists.at(oldlist) = next;
    }

    // std::cout<< "insert " << i << " into newlist" << std::endl;
    float head = pq->lists.at(newlist);
    // std::cout << " head = " << head << std::endl;
    if(head > -1){
        DegreePQElement* elem_head = new DegreePQElement();
        elem_head->prev = i;
        elem_head->next = pq->elems.at(head)->next;
        elem_head->key = pq->elems.at(head)->key;
        pq->elems.at(head) = elem_head;
    }
    pq->lists.at(newlist) = i;

    DegreePQElement* elem_i = new DegreePQElement();
    elem_i->prev = -1;
    elem_i->next = head;
    elem_i->key = newkey;
    pq->elems.at(i) = elem_i;
    // std::cout << "moved in the PQ" << std::endl;
    return;
}

void PriorityPreconditioner::DegreePQDec(DegreePQ* pq, int i){
    // std::cout << "dec in the PQ " << i << std::endl;
    float n = pq->n;
    float deg_i = pq->elems.at(i)->key;
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
        DegreePQElement* elem_i = new DegreePQElement();
        elem_i->prev = pq->elems.at(i)->prev;
        elem_i->next = pq->elems.at(i)->next;
        elem_i->key = pq->elems.at(i)->key - 1;
        pq->elems.at(i) = elem_i;
    }
    // std::cout << "deced in the PQ" << std::endl;
    return;
}

void PriorityPreconditioner::DegreePQInc(DegreePQ* pq, int i){
    // std::cout << "inc in the PQ" << std::endl;
    float n = pq->n;
    float deg_i = pq->elems.at(i)->key;
    int oldlist = deg_i <= n ? deg_i : n + int(deg_i/n);
    int newlist = deg_i+1 <= n ? deg_i+1 : n + int((deg_i+1)/n);
    // std::cout << " deg_i = " << deg_i << std::endl;
    if(oldlist != newlist){
        DegreePQMove(pq, i, deg_i+1, oldlist, newlist);
    }else{
        DegreePQElement* elem_i = new DegreePQElement();
        elem_i->prev = pq->elems.at(i)->prev;
        elem_i->next = pq->elems.at(i)->next;
        elem_i->key = pq->elems.at(i)->key + 1;
        pq->elems.at(i) = elem_i;
    }
    // std::cout << "inced in the PQ" << std::endl;
    return;
}

float PriorityPreconditioner::getColumnLength(PriorityMatrix* pmat, int i, std::vector<PriorityElement*>* colspace){
    // std::cout << "get col length " << i << std::endl;
    PriorityElement* ll = pmat->cols[i];
    float len = 0;
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

float PriorityPreconditioner::compressColumn(PriorityMatrix* a, std::vector<PriorityElement*>* colspace, float len, DegreePQ* pq){
    // std::cout << "Compressing col of len = " << len << std::endl;

    std::sort(colspace->begin(), colspace->begin()+len,
         [](PriorityElement* a, PriorityElement* b) {return a->row < b->row; });

    float ptr = PTR_RESET;
    float currow = -1;

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
         [](PriorityElement* a, PriorityElement* b) {return a->val < b->val; });
    //  std::cout << "Compressed col to len = " << ptr+1 << std::endl;
    return ptr+1;
}


LDLi* PriorityPreconditioner::getLDLi(){
    PriorityMatrix* a = _pmat;
    float n = a->n;
    LDLi* ldli = new LDLi();
    ldli->col = std::vector<float>();
    ldli->colptr = std::vector<float>();
    ldli->rowval = std::vector<float>();
    ldli->fval = std::vector<float>();

    float ldli_row_ptr = 0;
    std::vector<float> d(n, 0.0);

    DegreePQ* pq = getDegreePQ(a->degs);

    float it = 1;
    std::vector<PriorityElement*>* colspace = new std::vector<PriorityElement*>();
    std::mt19937_64 rand_generator;
    std::uniform_real_distribution<float> u_distribution(0, 1);
    while(it < n){
        // std::cout << "looop " << it << std::endl;
        float i = DegreePQPop(pq);

        ldli->col.push_back(i);
        ldli->colptr.push_back(ldli_row_ptr);

        it += 1;

        float len = getColumnLength(a, i, colspace);
        len = compressColumn(a, colspace, len, pq);
        float csum = 0;
        std::vector<float> cumspace;
        std::vector<float> vals;
        for(int ii = 0; ii < len ; ii++){
            vals.push_back(colspace->at(ii)->val);
            csum += colspace->at(ii)->val;
            cumspace.push_back(csum);
        }
        float wdeg = csum;
        float colScale = 1;

        for(int joffset = 0; joffset < len-1; joffset++){
            PriorityElement* ll = colspace->at(joffset);
            float w = vals.at(joffset) * colScale;
            float j = ll->row;
            PriorityElement* revj = ll->reverse;
            float f = float(w)/wdeg;
            vals.at(joffset) = 0;

            float u_r = u_distribution(rand_generator);
            float r = u_r*(csum  - cumspace[joffset]) + cumspace[joffset];
            float koff = -1;
            for(int k_i = 0; k_i < len; k_i++){
                if(cumspace[k_i]>r){
                    koff = k_i;
                    break;
                }
            }
            float k = colspace->at(koff)->row;

            DegreePQInc(pq, k);

            float newEdgeVal = f*(1-f)*wdeg;

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
            float w = vals.at(len-1)*colScale;
            float j = ll->row;
            PriorityElement* revj = ll->reverse;
            if(it < n){
                DegreePQDec(pq, j);
            }
            revj->val = 0;

            ldli->rowval.push_back(j);
            ldli->fval.push_back(1.0);
            ldli_row_ptr += 1;

            d[i] = w;
        }
    }
    ldli->colptr.push_back(ldli_row_ptr);
    ldli->d = d;
    // std::cout << " J OFFSETS = " << joffsets << std::endl;
    return ldli;
}