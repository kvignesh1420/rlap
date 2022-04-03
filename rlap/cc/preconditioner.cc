#include <iostream>
#include <vector>
#include <functional>
#include <random>
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
                float jhead = _ordmat->cols[j];

                OrderedElement* ord = new OrderedElement();
                ord->row = k;
                ord->next = jhead;
                ord->val = newEdgeVal;
                _ordmat->elements[ColumnElement.ptr] = ord;

                _ordmat->cols[j] = ColumnElement.ptr;
            } else{
                float khead = _ordmat->cols[k];

                OrderedElement* ord = new OrderedElement();
                ord->row = j;
                ord->next = khead;
                ord->val = newEdgeVal;
                _ordmat->elements[ColumnElement.ptr] = ord;

                _ordmat->cols[k] = ColumnElement.ptr;
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