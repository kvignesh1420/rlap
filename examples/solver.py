"""
Python example for the solver

NOTE:
Please run the data/generator.py script with python3 to generate
the data files.
"""

import numpy as np
from rlap import ApproximateCholesky

def main():
    N = 1000000
    filename = "data/grid100.tsv"
    # initialize the factorizer
    fact = ApproximateCholesky(filename=filename, nrows=N, ncols=N)
    # retrieve the laplacian
    L = fact.get_laplacian()
    # generate a random ground truth x_gt
    x_gt = np.random.rand(N)
    # calculate the respective b
    b = L*x_gt
    # estimate the ground truth by pcg using the preconditioner from
    # the factorizer
    x = fact.solve(b)

if __name__ == "__main__":
    main()