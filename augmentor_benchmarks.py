"""
Benchmark the resource consumption of graph augmentors.
"""

import time
import argparse
from memory_profiler import memory_usage, profile
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
from GCL.augmentors.augmentor import Graph, Augmentor
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import sort_edge_index, degree, to_networkx, to_undirected
from torch_geometric.data import Data
from torch_sparse import coalesce
from torch_scatter import scatter
import networkx as nx

from torch_geometric.datasets import Planetoid, Coauthor, WikiCS, SNAPDataset, TUDataset
import numpy as np
from rlap import ApproximateCholesky


def coalesce_edge_index(edge_index: torch.Tensor, edge_weights = None):
    num_edges = edge_index.size()[1]
    num_nodes = edge_index.max().item() + 1
    edge_weights = edge_weights if edge_weights is not None else torch.ones((num_edges,), dtype=torch.float32, device=edge_index.device)

    return coalesce(edge_index, edge_weights, m=num_nodes, n=num_nodes)


def add_edge(edge_index: torch.Tensor, ratio: float):
    num_edges = edge_index.size()[1]
    num_nodes = edge_index.max().item() + 1
    num_add = int(num_edges * ratio)

    new_edge_index = torch.randint(0, num_nodes - 1, size=(2, num_add)).to(edge_index.device)
    edge_index = torch.cat([edge_index, new_edge_index], dim=1)
    edge_index = sort_edge_index(edge_index)
    return coalesce_edge_index(edge_index)[0]

class EdgeAdding(Augmentor):
    def __init__(self, pe: float):
        super(EdgeAdding, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index = add_edge(edge_index, ratio=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class rLap(A.Augmentor):
    def __init__(self, t):
        super(rLap, self).__init__()
        self.t = t
    
    def augment(self, g):
        x, edge_index, edge_weights = g.unfold()
        num_nodes = edge_index.max().item() + 1
        _edge_weights = edge_weights
        if _edge_weights is None:
            edge_weights = torch.ones((1, edge_index.shape[1])).to(edge_index.device)
        edge_info = torch.concat((edge_index, edge_weights), dim=0).t()
        ac = ApproximateCholesky()
        ac.setup(edge_info.to("cpu"), num_nodes, num_nodes, "random")
        sparse_edge_info = ac.get_schur_complement(self.t)
        sampled_edge_index = torch.Tensor(sparse_edge_info[:,:2]).long().t().to(edge_index.device)
        sampled_edge_weights = torch.Tensor(sparse_edge_info[:,-1]).t().to(edge_index.device)
        del ac
        del sparse_edge_info
        return A.Graph(x=x, edge_index=sampled_edge_index, edge_weights=sampled_edge_weights)

# Adaptive Augmentors from (GCA):
# https://github.com/CRIPAC-DIG/GCA


def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x

def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return sel_mask


def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights


def evc_drop_weights(data):
    evc = eigenvector_centrality(data)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())


class EdgeDroppingDegree(A.Augmentor):
    def __init__(self, p, threshold):
        super(EdgeDroppingDegree, self).__init__()
        self.p = p
        self.threshold = threshold

    def augment(self, g):
        x, edge_index, edge_weights = g.unfold()
        drop_weights = degree_drop_weights(edge_index=edge_index)
        mask = drop_edge_weighted(
            edge_index=edge_index,
            edge_weights=drop_weights,
            p=self.p,
            threshold=self.threshold
        )
        sampled_edge_index = edge_index[:, mask]
        sampled_edge_weights = edge_weights[mask] if edge_weights is not None else edge_weights

        return A.Graph(x=x, edge_index=sampled_edge_index, edge_weights=sampled_edge_weights)


class EdgeDroppingPR(A.Augmentor):
    def __init__(self, p, threshold):
        super(EdgeDroppingPR, self).__init__()
        self.p = p
        self.threshold = threshold

    def augment(self, g):
        x, edge_index, edge_weights = g.unfold()
        drop_weights = pr_drop_weights(edge_index=edge_index)
        mask = drop_edge_weighted(
            edge_index=edge_index,
            edge_weights=drop_weights,
            p=self.p,
            threshold=self.threshold
        )
        sampled_edge_index = edge_index[:, mask]
        sampled_edge_weights = edge_weights[mask] if edge_weights is not None else edge_weights

        return A.Graph(x=x, edge_index=sampled_edge_index, edge_weights=sampled_edge_weights)



class EdgeDroppingEVC(A.Augmentor):
    def __init__(self, p, threshold):
        super(EdgeDroppingEVC, self).__init__()
        self.p = p
        self.threshold = threshold

    def augment(self, g):
        x, edge_index, edge_weights = g.unfold()
        data = Data(x=x, edge_index=edge_index, edge_weights=edge_weights)
        drop_weights = evc_drop_weights(data=data)
        mask = drop_edge_weighted(
            edge_index=edge_index,
            edge_weights=drop_weights,
            p=self.p,
            threshold=self.threshold
        )
        sampled_edge_index = edge_index[:, mask]
        sampled_edge_weights = edge_weights[mask] if edge_weights is not None else edge_weights

        return A.Graph(x=x, edge_index=sampled_edge_index, edge_weights=sampled_edge_weights)


@profile()
def benchmark_memory(aug, data):
    aug(data.x, data.edge_index, data.edge_weight)

def benchmark_latency(aug, data):
    start = time.time()
    aug(data.x, data.edge_index, data.edge_weight)
    end = time.time()
    print("\nDURATION: {} sec\n".format(end-start))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('augmentor', type=str)
    parser.add_argument('dataset', type=str)
    args = parser.parse_args()
    print(args)

    device = torch.device('cpu')
    path = osp.join(osp.expanduser('~'), 'datasets')
    datasets = {
        "CORA": lambda: Planetoid(path, name='Cora', transform=T.NormalizeFeatures()),
        "WIKI-CS": lambda: WikiCS(path, transform=T.NormalizeFeatures()),
        "COAUTHOR-CS": lambda: Coauthor(path, name="CS", transform=T.NormalizeFeatures()),
        "COAUTHOR-PHY": lambda: Coauthor(path, name="Physics", transform=T.NormalizeFeatures()),
    }

    dataset = datasets[args.dataset]()
    data = dataset[0].to(device)
    num_nodes = data.edge_index.max().item() + 1
    fraction = 0.5
    usages = {}
    augmentors = {
        "rLap": A.Compose([rLap(int(fraction*num_nodes))]),
        "EdgeAddition": A.Compose([EdgeAdding(pe=0.5)]),
        "EdgeDropping": A.Compose([A.EdgeRemoving(pe=0.5)]),
        "EdgeDroppingDegree": A.Compose([EdgeDroppingDegree(p=0.5, threshold=0.7)]),
        "EdgeDroppingPR": A.Compose([EdgeDroppingPR(p=0.5, threshold=0.7)]),
        "EdgeDroppingEVC": A.Compose([EdgeDroppingEVC(p=0.5, threshold=0.7)]),
        "NodeDropping": A.Compose([A.NodeDropping(pn=0.5)]),
        "RandomWalkSubgraph": A.Compose([A.RWSampling(num_seeds=int(fraction*num_nodes), walk_length=10)]),
        "PPRDiffusion": A.Compose([A.PPRDiffusion(alpha=0.2, use_cache=False)]),
        "MarkovDiffusion": A.Compose([A.MarkovDiffusion(alpha=0.2, use_cache=False)]),
        # For this combination, leverage the fact that we need same set of nodes across views.
        # We can thus, significantly compress the size of 'x' being passed to PPR.
        "rLap+PPRDiffusion": A.Compose([rLap(int(fraction*num_nodes)), A.PPRDiffusion(alpha=0.2, use_cache=False)]),
        "rLap+MarkovDiffusion": A.Compose([rLap(int(fraction*num_nodes)), A.MarkovDiffusion(alpha=0.2, use_cache=False)]),
    }
    aug = augmentors[args.augmentor]

    benchmark_memory(aug, data)
    benchmark_latency(aug, data)