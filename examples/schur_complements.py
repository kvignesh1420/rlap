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
a = ApproximateCholesky()
n = 1965206
a.setup(edge_info, n, n, "order")
res = a.get_schur_complement(100000)
print(res.shape)
b = ApproximateCholesky()
b.setup(edge_info, n, n, "coarsen")
res_b = b.get_schur_complement(100000)
print(res_b.shape)