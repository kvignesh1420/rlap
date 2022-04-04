# rlap : A randomized laplacian solver for large linear systems

This work is inspired by the [approximate gaussian elemination](https://arxiv.org/abs/1605.02353) methods for solving laplacian system of equations using randomization techniques.

### Usage

`rlap` can be seamlessly integrated with your application using bazel targets. Set it up as an external dependency and get started.
#### Setup

_A helper script has been provided to generate connected and grid graphs of various sizes for experimentation and analysis._

```bash
$ cd data
$ pip3 install -r requirements.txt
$ python3 generate.py
```

Also, you would need the `Eigen` library for representing sparse matrices in column compressed storage format and dealing with matrices or vectors in general.

#### Example

A laplacian system of equations can be solved using `rlap` as follows:

```c++
#include <iostream>
#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/Eigen/SparseCholesky"
#include "rlap/cc/reader.h"
#include "rlap/cc/factorizers.h"

int main(){
    // read the tsv adjacency matrix
    int N = 125000; // number of rows/cols
    std::string filepath = "data/grid50.tsv";
    Reader* r = new TSVReader(filepath, N, N);
    Eigen::SparseMatrix<float>* A = r->Read();

    // initialize the factorizer
    ApproximateCholesky fact = ApproximateCholesky(A);

    // retrieve the laplacian of the graph
    Eigen::SparseMatrix<float> L = fact.getLaplacian();
    // generate random b vector
    Eigen::VectorXf b = Eigen::VectorXf::Random(BLOCK_SIZE);
    // normalize b
    float b_mean = b.mean();
    Eigen::VectorXf b_m = b - Eigen::VectorXf::Ones(b.size())*b_mean;

    // solve Lx = b_m with pre-conditioned conjugate descent
    x = fact.solve(b_m);

    std::cout << "Approximation error = " << (L*x - b_m).norm()/b_m.norm() << std::endl;

    return 0;

}

```

Additional example(s) are available in `examples/` directory. They can be built using:
```
$ bazel build //examples:all
```

for instance, the `examples/solver` target can now be executed as:
```
$ bazel-bin/examples/solver
```
