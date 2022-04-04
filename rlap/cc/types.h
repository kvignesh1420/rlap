#ifndef RLAP_CC_TYPES_H
#define RLAP_CC_TYPES_H

#include <vector>

typedef struct PriorityElement{
    // Represents an element of a matrix along with the
    // information about the next and previous elements in
    // it's column. The operations pertaining to this struct
    // are optimized for sparse matrices with compressed column storage.

    // row number of the element
    float row;

    // nnz value of the element
    float val;

    // pointer to the next element in the column
    PriorityElement* next;

    // pointer to the reverse element, i.e an element from which a reverse
    // traversal of the column is possible.
    PriorityElement* reverse;

    PriorityElement(){
        row = 0;
        val = 0;
        next = this;
        reverse = this;
    }
    PriorityElement(float _row, float _val, PriorityElement* _next, PriorityElement* _reverse){
        row = _row;
        val = _val;
        next = _next;
        reverse = _reverse;
    }
    PriorityElement(float _row, float _val){
        row = _row;
        val = _val;
        next = this;
        reverse = this;
    }
    PriorityElement(float _row, float _val, PriorityElement* _next){
        row = _row;
        val = _val;
        next = _next;
        reverse = this;
    }

} PriorityElement;

typedef struct PriorityMatrix{
    // Represents the original matrix as a matrix of `PriorityElements`.
    // along with information on the degrees of the graph nodes.

    // number of columns = number of nodes in the graph
    float n;

    // degrees of the node(s) = number of nnz values in column(s)
    std::vector<float> degs;

    // pointers to `PriorityElement` which start the column
    std::vector<PriorityElement*> cols;

    // pointers to `PriorityElement` in the column major order.
    std::vector<PriorityElement*> lles;
} PriorityMatrix;

//                Structure for the priority queue

// The `DegreePQ` struct maintains a vector of pointers to `DegreePQElement`.
// The column number (which indicates node number in the graph) are grouped
// based on their degree into linked lists. For example, if columns/nodes 1, 4, 5
// have degree 3, then 1 <-> 4 <-> 5 will be a doubly linked list that is maintained
// and the value 5, which is the last (from left to right) is stored in `DegreePQ.lists`
// at index 5. NOTE: even though we index starting from 0 in c++, this shouldn't be an
// issue as the nodes with degree 0 have no edges to sample from.

typedef struct DegreePQElement{
    // Represents an element in the priority queue for degree
    // aware sampling of edges.

    // index of the `DegreePQElement` in `DegreePQ.elems` which is
    // placed before the current `DegreePQElement` in the queue for
    // degree d.
    float prev;

    // index of the `DegreePQElement` in `DegreePQ.elems` which is
    // placed after the current `DegreePQElement` in the queue for
    // degree d.
    float next;

    // value of the `DegreePQElement`.
    float key;
} DegreePQElement;

typedef struct DegreePQ{
    // Represents a priority queue structure for grouping nodes
    // based on degrees.

    // a vector of pointers to the `DegreePQElement`s, i.e the
    // elements in the priority queue.
    std::vector<DegreePQElement*> elems;

    // a vector whose index represents the degree of nodes in the graph.
    // by default `lists` is filled with -1 and the index of the
    // `DegreePQElement` with a certain degree (d) that will be eliminated
    // is stored at index (d) in `lists`.
    std::vector<float> lists;

    // represents the min-degree of all degrees of the nodes in the graph, i.e
    // the column with the least number of nnz values.
    float minlist;

    // number of items in the queue
    float nitems;

    // total number of nodes in the graph = number of columns in the graph.
    float n;
} DegreePQ;


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