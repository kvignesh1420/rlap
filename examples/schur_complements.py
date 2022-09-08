import torch
import numpy as np
from rlap import ApproximateCholesky

edge_info = []
with open("data/roadNet-CA_adj.tsv", "r") as f:
    while True:
        a = f.readline()
        if not a:
            break
        a = a.split("\t")
        a[-1] = a[-1].replace("\n","")
        a = list(map(lambda x: int(x), a))
        a[0] -= 1
        a[1] -= 1
        edge_info.append(a)
edge_info = torch.Tensor(np.array(edge_info))
n = 1965206
frac = 0.3
remaining_n = int(n * frac)

for strat in ["order", "coarsen", "random", "degree"]:
    a = ApproximateCholesky()
    a.setup(edge_info, n, n, strat)
    res_a = a.get_schur_complement(remaining_n)
    print("STRAT: {} SC SHAPE: {}".format(strat, res_a.shape))
