"""
Python example for the solver

NOTE:
Please run the data/generator.py script with python3 to generate
the data files.
"""

import time
import numpy as np
from rlap import ApproximateCholesky
from scipy.sparse import linalg


def main():
    N = 1000000
    filename = "data/grid100.tsv"
    # initialize the factorizer
    fact = ApproximateCholesky(filename=filename, nrows=N, ncols=N, pre="order")
    # retrieve the laplacian
    L = fact.get_laplacian()
    # generate a random ground truth x_gt
    x_gt = np.random.rand(N)
    # calculate the respective b
    b = L*x_gt
    # estimate the ground truth by pcg using the preconditioner from
    # the factorizer
    x = fact.solve(b)
    s = time.time()
    x_t = linalg.bicg(L, b, tol=1e-12, maxiter=3000)
    e = time.time()
    print(e-s)


if __name__ == "__main__":
    main()