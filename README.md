## rlap : A randomized laplacian solver for large linear systems

`rlap` is a laplacian system solver in C++/Python that is inspired by the [approximate gaussian elemination](https://arxiv.org/abs/1605.02353) methods using randomization techniques. It generates a pre-conditioner for the laplacians and uses the iterative conjugate descent algorithm to converge to a tolerable solution **within seconds**.

The [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) library is used for representing sparse matrices and vectors which aids in efficient traversal and indexing into matrices. Additionally, the fast pre-conditioning techniques and the relevant data structures are inspired from the [Laplacians.jl](https://github.com/danspielman/Laplacians.jl) effort.

### Usage

`rlap` can be seamlessly integrated with your c++ application using bazel targets. Set it up as an external dependency and get started.
#### Setup

_A helper script has been provided to generate connected and grid graphs of various sizes for experimentation and analysis._

```bash
$ cd data
$ pip3 install -r requirements.txt
$ python3 generate.py
```

#### Example

A laplacian system of equations can be solved using `rlap` as follows:

```python
import numpy as np
from rlap import ApproximateCholesky

dim = 1000000
filename = "data/grid100.tsv"
# initialize the factorizer
fact = ApproximateCholesky(filename=filename, nrows=dim, ncols=dim)
# retrieve the laplacian
L = fact.get_laplacian()
# generate a random ground truth x_gt
x_gt = np.random.rand(dim)
# calculate the respective b
b = L * x_gt

# estimate the ground truth by pcg using the preconditioner from
# the factorizer
x = fact.solve(b)
```

A similar example in C++ would look as follows:

```c++
#include <iostream>
#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/Eigen/SparseCholesky"
#include "rlap/cc/reader.h"
#include "rlap/cc/factorizers.h"

int main(){
    // read the tsv adjacency matrix
    int N = 1000000; // number of rows/cols
    std::string filepath = "data/grid100.tsv";
    Reader* r = new TSVReader(filepath, N, N);
    Eigen::SparseMatrix<double>* A = r->Read();
    // initialize the factorizer
    ApproximateCholesky fact = ApproximateCholesky(A);
    // retrieve the laplacian of the graph
    Eigen::SparseMatrix<double> L = fact.getLaplacian();
    // generate a random ground truth
    Eigen::VectorXd x_t = Eigen::VectorXd::Random(N);
	// generate target based on random x_t
	Eigen::VectorXd b = L * x_t;
    // solve Lx = b with pre-conditioned conjugate descent
    x = fact.solve(b);

    return 0;
}

```

Detailed example(s) are available in `examples/` directory. They can be built using:
```
$ bazel build //examples:all
```

for instance, the `examples/solver` c++ target can now be executed as:
```
$ bazel-bin/examples/solver
```

For memory leak detection with valgrind, add the `build --cxxopt="-ggdb3"` setting in `.bazelrc` and run as follows:

```
$ bazel build //rlap:all
$ bazel build //examples:all

$ valgrind --leak-check=full \\
         --show-leak-kinds=all \\
         --track-origins=yes \\
         --verbose \\
         --log-file=valgrind-out.txt \\
         ./bazel-bin/examples/solver
```

