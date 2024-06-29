"""
Benchmark the resource consumption of graph augmentors.
"""

import time
import argparse
from memory_profiler import profile
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import compute_ppr
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import (
    sort_edge_index,
    degree,
    to_networkx,
    to_undirected,
    subgraph,
)
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_sparse import coalesce
from torch_scatter import scatter
import networkx as nx
from torch_geometric.datasets import Planetoid, Coauthor, TUDataset, Amazon
import rlap


def coalesce_edge_index(edge_index: torch.Tensor, edge_weights=None):
    num_edges = edge_index.size()[1]
    num_nodes = edge_index.max().item() + 1
    edge_weights = (
        edge_weights
        if edge_weights is not None
        else torch.ones((num_edges,), dtype=torch.float32, device=edge_index.device)
    )

    return coalesce(edge_index, edge_weights, m=num_nodes, n=num_nodes)


def add_edge(edge_index: torch.Tensor, ratio: float):
    num_edges = edge_index.size()[1]
    num_nodes = edge_index.max().item() + 1
    num_add = int(num_edges * ratio)

    new_edge_index = torch.randint(0, num_nodes - 1, size=(2, num_add)).to(
        edge_index.device
    )
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
    def __init__(self, frac, o_v="random", o_n="asc"):
        super(rLap, self).__init__()
        self.frac = frac
        self.o_v = o_v
        self.o_n = o_n

    def augment(self, g):
        x, edge_index, edge_weights = g.unfold()
        num_nodes = edge_index.max().item() + 1
        self.num_remove = int(self.frac * num_nodes)
        _edge_weights = edge_weights
        if _edge_weights is None:
            edge_weights = torch.ones((1, edge_index.shape[1])).to(edge_index.device)
        edge_info = torch.concat((edge_index, edge_weights), dim=0).t()
        sparse_edge_info = rlap.ops.approximate_cholesky(
            edge_info=edge_info.to("cpu"),
            num_nodes=num_nodes,
            num_remove=self.num_remove,
            o_v=self.o_v,
            o_n=self.o_n,
        )
        sampled_edge_index = (
            torch.Tensor(sparse_edge_info[:, :2]).long().t().to(edge_index.device)
        )
        # NOTE: uncomment the following and update the return statement
        #  to incorporate edge-weight information (if needed).

        # sampled_edge_weights = torch.Tensor(sparse_edge_info[:,-1]).t().to(edge_index.device)
        del sparse_edge_info
        return A.Graph(x=x, edge_index=sampled_edge_index, edge_weights=None)


class rLapPPRDiffusion(A.Augmentor):
    def __init__(
        self,
        frac,
        o_v="random",
        o_n="asc",
        alpha=0.2,
        eps=1e-4,
        use_cache=True,
        refresh_cache_freq=50,
    ):
        super(rLapPPRDiffusion, self).__init__()
        self.frac = frac
        self.o_v = o_v
        self.o_n = o_n
        self.alpha = alpha
        self.eps = eps
        self._cache = None
        self.use_cache = use_cache
        self.refresh_cache_freq = refresh_cache_freq
        self.refresh_cache_counter = 0

    def augment(self, g):
        if (
            self._cache is not None
            and self.use_cache
            and self.refresh_cache_counter < self.refresh_cache_freq
        ):
            self.refresh_cache_counter += 1
            return self._cache
        x, edge_index, edge_weights = g.unfold()
        num_nodes = edge_index.max().item() + 1
        self.num_remove = int(self.frac * num_nodes)
        _edge_weights = edge_weights
        if _edge_weights is None:
            edge_weights = torch.ones((1, edge_index.shape[1])).to(edge_index.device)
        edge_info = torch.concat((edge_index, edge_weights), dim=0).t()
        sparse_edge_info = rlap.ops.approximate_cholesky(
            edge_info=edge_info.to("cpu"),
            num_nodes=num_nodes,
            num_remove=self.num_remove,
            o_v=self.o_v,
            o_n=self.o_n,
        )
        sampled_edge_index = (
            torch.Tensor(sparse_edge_info[:, :2]).long().t().to(edge_index.device)
        )
        sampled_edge_weights = (
            torch.Tensor(sparse_edge_info[:, -1]).t().to(edge_index.device)
        )

        del sparse_edge_info
        sc_subgraph_nodes = torch.unique(sampled_edge_index, sorted=True)
        sc_subgraph_edge_index, sc_subgraph_edge_weights = subgraph(
            subset=sc_subgraph_nodes,
            edge_index=sampled_edge_index,
            edge_attr=sampled_edge_weights,
            relabel_nodes=True,
        )

        diffused_edge_index, diffused_edge_weights = compute_ppr(
            sc_subgraph_edge_index,
            sc_subgraph_edge_weights,
            alpha=self.alpha,
            eps=self.eps,
            ignore_edge_attr=False,
            add_self_loop=False,
        )
        diffused_edge_index = sc_subgraph_nodes[diffused_edge_index]
        res = A.Graph(
            x=x, edge_index=diffused_edge_index, edge_weights=diffused_edge_weights
        )
        self._cache = res
        self.refresh_cache_counter = 0
        return res


class PPRDiffusionSubGraph(Augmentor):
    def __init__(
        self,
        alpha: float = 0.2,
        eps: float = 1e-4,
        use_cache: bool = True,
        add_self_loop: bool = True,
        sub_graph_size=8192,
    ):
        super(PPRDiffusionSubGraph, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self._cache = None
        self.use_cache = use_cache
        self.add_self_loop = add_self_loop
        self.sub_graph_size = sub_graph_size

    def augment(self, g: Graph) -> Graph:
        if self._cache is not None and self.use_cache:
            return self._cache
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = compute_ppr(
            edge_index,
            edge_weights,
            alpha=self.alpha,
            eps=self.eps,
            ignore_edge_attr=False,
            add_self_loop=self.add_self_loop,
        )

        node_indices = torch.unique(edge_index)
        num_nodes = node_indices.shape[0]
        perm = torch.randperm(num_nodes)
        node_indices = node_indices[perm]
        batch_nodes = node_indices[: self.sub_graph_size]

        edge_index, edge_weights = subgraph(batch_nodes, edge_index, edge_weights)
        res = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        self._cache = res
        return res


def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes,)).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce="sum")

        x = (1 - damp) * x + damp * agg_msg

    return x


def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph, tol=1e-05)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.0):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(
        edge_weights < threshold, torch.ones_like(edge_weights) * threshold
    )
    sel_mask = torch.bernoulli(1.0 - edge_weights).to(torch.bool)

    return sel_mask


def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def pr_drop_weights(edge_index, aggr: str = "sink", k: int = 10):
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == "sink":
        s = s_col
    elif aggr == "source":
        s = s_row
    elif aggr == "mean":
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
            threshold=self.threshold,
        )
        sampled_edge_index = edge_index[:, mask]
        sampled_edge_weights = (
            edge_weights[mask] if edge_weights is not None else edge_weights
        )

        return A.Graph(
            x=x, edge_index=sampled_edge_index, edge_weights=sampled_edge_weights
        )


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
            threshold=self.threshold,
        )
        sampled_edge_index = edge_index[:, mask]
        sampled_edge_weights = (
            edge_weights[mask] if edge_weights is not None else edge_weights
        )

        return A.Graph(
            x=x, edge_index=sampled_edge_index, edge_weights=sampled_edge_weights
        )


class EdgeDroppingEVC(A.Augmentor):
    def __init__(self, p, threshold):
        super(EdgeDroppingEVC, self).__init__()
        self.p = p
        self.threshold = threshold

    def augment(self, g):
        x, edge_index, edge_weights = g.unfold()
        device = torch.device("cuda")
        data = Data(x=x, edge_index=edge_index, edge_weights=edge_weights).to(device)
        drop_weights = evc_drop_weights(data=data)
        mask = drop_edge_weighted(
            edge_index=edge_index,
            edge_weights=drop_weights,
            p=self.p,
            threshold=self.threshold,
        )
        sampled_edge_index = edge_index[:, mask]
        sampled_edge_weights = (
            edge_weights[mask] if edge_weights is not None else edge_weights
        )

        return A.Graph(
            x=x, edge_index=sampled_edge_index, edge_weights=sampled_edge_weights
        )


@profile()
def benchmark_node_memory(aug, data):
    aug(data.x, data.edge_index, data.edge_weight)


def benchmark_node_latency(aug, data):
    start = time.time()
    aug(data.x, data.edge_index, data.edge_weight)
    end = time.time()
    print("\nDURATION: {} sec\n".format(end - start))


@profile()
def benchmark_graph_memory(aug, dataloader):
    for data in dataloader:
        data = data.to(device)
        aug(data.x, data.edge_index, data.edge_weight)


def benchmark_graph_latency(aug, dataloader):
    duration = 0
    for data in dataloader:
        data = data.to(device)
        start = time.time()
        aug(data.x, data.edge_index, data.edge_weight)
        end = time.time()
        duration += end - start
    print("\nDURATION: {} sec\n".format(duration))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    parser.add_argument("augmentor", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("device", type=str)
    args = parser.parse_args()
    print(args)

    device = torch.device(args.device)
    path = osp.join(osp.expanduser("~"), "datasets")
    datasets = {
        # node
        "CORA": lambda: Planetoid(path, name="Cora", transform=T.NormalizeFeatures()),
        "AMAZON-PHOTO": lambda: Amazon(
            path, name="Photo", transform=T.NormalizeFeatures()
        ),
        "PUBMED": lambda: Planetoid(
            path, name="PubMed", transform=T.NormalizeFeatures()
        ),
        "COAUTHOR-CS": lambda: Coauthor(
            path, name="CS", transform=T.NormalizeFeatures()
        ),
        "COAUTHOR-PHY": lambda: Coauthor(
            path, name="Physics", transform=T.NormalizeFeatures()
        ),
        # graph
        "PROTEINS": lambda: TUDataset(path, name="PROTEINS_full"),
        "IMDB-BINARY": lambda: TUDataset(path, name="IMDB-BINARY"),
        "IMDB-MULTI": lambda: TUDataset(path, name="IMDB-MULTI"),
        "MUTAG": lambda: TUDataset(path, name="MUTAG"),
        "NCI1": lambda: TUDataset(path, name="NCI1"),
    }

    fraction = 0.5
    dataset = datasets[args.dataset]()
    if args.task == "graph":
        dataloader = DataLoader(dataset, batch_size=128)
        num_seeds = 1000
    elif args.task == "node":
        data = dataset[0].to(device)
        num_nodes = data.edge_index.max().item() + 1
        num_seeds = int(fraction * num_nodes)

    usages = {}
    augmentors = {
        "rLap": A.Compose([rLap(fraction)]),
        "EdgeAddition": A.Compose([EdgeAdding(pe=fraction)]),
        "EdgeDropping": A.Compose([A.EdgeRemoving(pe=fraction)]),
        "EdgeDroppingDegree": A.Compose(
            [EdgeDroppingDegree(p=fraction, threshold=0.7)]
        ),
        "EdgeDroppingPR": A.Compose([EdgeDroppingPR(p=fraction, threshold=0.7)]),
        "EdgeDroppingEVC": A.Compose([EdgeDroppingEVC(p=fraction, threshold=0.7)]),
        "NodeDropping": A.Compose([A.NodeDropping(pn=fraction)]),
        "RandomWalkSubgraph": A.Compose(
            [A.RWSampling(num_seeds=num_seeds, walk_length=10)]
        ),
        "PPRDiffusion": A.Compose([A.PPRDiffusion(alpha=0.2, use_cache=False)]),
        "MarkovDiffusion": A.Compose([A.MarkovDiffusion(alpha=0.2, use_cache=False)]),
    }
    aug = augmentors[args.augmentor]

    if args.task == "node":
        if args.device == "cpu":
            benchmark_node_memory(aug, data)
        benchmark_node_latency(aug, data)

    if args.task == "graph":
        if args.device == "cpu":
            benchmark_graph_memory(aug, dataloader)
        benchmark_graph_latency(aug, dataloader)
