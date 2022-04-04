#ifndef RLAP_CC_TYPES_H
#define RLAP_CC_TYPES_H

#include <vector>

typedef struct LLp{
    float row;
    float val;
    LLp* next;
    LLp* reverse;

    LLp(){
        row = 0;
        val = 0;
        next = this;
        reverse = this;
    }
    LLp(float _row, float _val, LLp* _next, LLp* _reverse){
        row = _row;
        val = _val;
        next = _next;
        reverse = _reverse;
    }
    LLp(float _row, float _val){
        row = _row;
        val = _val;
        next = this;
        reverse = this;
    }
    LLp(float _row, float _val, LLp* _next){
        row = _row;
        val = _val;
        next = _next;
        reverse = this;
    }

} LLp;

typedef struct LLMatp
{
    float n;
    std::vector<float> degs;
    std::vector<LLp*> cols;
    std::vector<LLp*> lles;
} LLMatp;

typedef struct ApproxCholPQElem{
    float prev;
    float next;
    float key;
} ApproxCholPQElem;

typedef struct ApproxCholPQ{
    std::vector<ApproxCholPQElem*> elems;
    std::vector<float> lists;
    float minlist;
    float nitems;
    float n;
} ApproxCholPQ;


typedef struct OrderedElement{
    // Represents an element of a matrix along with the
    // information about the order in which it appears in
    // it's column. The operations pertaining to this struct
    // are optimized for sparse matrices with compressed column storage.

    // represents the row number of the nnz element
    float row;

    // points to the next nnz element in the column
    float next;

    // the nnz value of the element
    float val;
} OrderedElement;

typedef struct OrderedMatrix{
    // Represents a given matrix in a column major order
    // by maintaining a vector of OrderedElement's and the
    // corresponding vector which maintains the indices of
    // OrderedElement's where a particular column starts.

    // number of columns
    float n;

    // a vector to maintain the indices of
    // OrderedElement's where a particular column starts.
    std::vector<float> cols;

    // a vector to maintain the OrderedElemnts of the matrix
    std::vector<OrderedElement*> elements;
} OrderedMatrix;

typedef struct ColumnElement{
    // Represents an element of a particular column when
    // dealing with the column in it's entirety instead of
    // the matrix itself.

    // NOTE: Since each OrderedElement represents an edge in the
    // graph between vertices, this struct enables us to compress
    // those multi-edges into a single edge and store as a
    // ColumnElement for column processing.

     // represents the row number of the nnz element
    float row;

    // points to the next nnz element in the column
    float ptr;

    // the nnz value of the element
    float val;
} ColumnElement;

typedef struct LDLi{
    // Represents the preconditioner pertaining to the
    // L * D * L^-1 operation while solving the linear system of
    // equations.

    // a vector of column numbers
    std::vector<float> col;

    // a vector to represent where in `rowval`,
    // a particular column starts.
    std::vector<float> colptr;

    // a vector of row numbers of nnz values
    std::vector<float> rowval;

    // the fractional weight of a particular edge
    // for approximating a clique.
    std::vector<float> fval;

    // a vector of diagonal elements of D
    std::vector<float> d;
} LDLi;

#endif