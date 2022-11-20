#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "rlap/cc/factorizers.h"

namespace py = pybind11;

PYBIND11_MODULE(_librlap, m){
    m.doc() = "api wrapper for the rlap c++ library.";
    py::class_<Factorizer>(m, "Factorizer");

    // the py::dynamic_attr() tag has been set to enable dynamic attributes on the
    // ApproximateCholesky class at the python layer. However, this can lead to a small
    // runtime cost as a __dict__ is now added to this python class, and the garbage
    // collection becomes a bit expensive.
    py::class_<ApproximateCholesky, Factorizer>(m, "ApproximateCholesky", py::dynamic_attr())
        // .def(py::init<std::string, int, int, std::string>(), py::arg("filename"), py::arg("nrows"), py::arg("ncols"), py::arg("o_v"))
        .def(py::init<>())
        .def("setup", &ApproximateCholesky::setup, py::arg("edge_info"), py::arg("nrows"), py::arg("ncols"), py::arg("o_v"), py::arg("o_n"),
            "setup the edge_info matrix and precondition the laplacian")
        .def("get_laplacian", &ApproximateCholesky::getLaplacian,
            "get the computed Laplacian")
        .def("solve", &ApproximateCholesky::solve, py::arg("b"),
            "solve the linear system of Lx = b, where L is the Laplacian"
            "computed from the adjacency matrix")
        .def("get_num_iters", &ApproximateCholesky::getNumIters,
            "return the number of iteration by pcg solver.")
        .def("get_sparsity_ratio", &ApproximateCholesky::getSparsityRatio,
            "return the ratio of number of preconditioned egdes to original edges.")
        .def("get_schur_complement", &ApproximateCholesky::getSchurComplement, py::arg("t"),
            "retrieve the schur complement after eliminating 't' vertices")
        .def("__repr__", 
            [](const ApproximateCholesky& a){ return "ApproximateCholesky()";}
        );
}
