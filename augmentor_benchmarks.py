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
from torch_geometric.utils import sort_edge_index
from torch_sparse import coalesce


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


class rLAP(A.Augmentor):
    def __init__(self, t):
        super(rLAP, self).__init__()
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

    # dataset = SNAPDataset(path, name="soc-Pokec", transform=T.NormalizeFeatures())
    # dataset = SNAPDataset(path, name="ego-facebook", transform=T.NormalizeFeatures())
    # dataset = TUDataset(path, name="REDDIT-BINARY", transform=T.NormalizeFeatures())

    dataset = datasets[args.dataset]()
    data = dataset[0].to(device)
    num_nodes = data.edge_index.max().item() + 1
    fraction = 0.5
    usages = {}
    augmentors = {
        "rLap": A.Compose([rLAP(int(fraction*num_nodes))]),
        "EdgeAddition": A.Compose([EdgeAdding(pe=0.5)]),
        "EdgeDropping": A.Compose([A.EdgeRemoving(pe=0.5)]),
        "NodeDropping": A.Compose([A.NodeDropping(pn=0.5)]),
        "RandomWalkSubgraph": A.Compose([A.RWSampling(num_seeds=int(fraction*num_nodes), walk_length=10)]),
        "PPRDiffusion": A.Compose([A.PPRDiffusion(alpha=0.2, use_cache=False)]),
        "MarkovDiffusion": A.Compose([A.MarkovDiffusion(alpha=0.2, use_cache=False)]),
    }
    aug = augmentors[args.augmentor]

    benchmark_memory(aug, data)
    benchmark_latency(aug, data)