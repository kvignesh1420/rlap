import torch as th
from torch_scatter import scatter
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_networkx, to_undirected
import networkx as nx
from GCL.augmentors import functional
import numpy as np
import dgl
import sys
sys.path.append('../bazel-bin/rlap')
sys.path.append('../')
from rlap import ApproximateCholesky


def random_aug(graph, x, feat_drop_rate, frac):
    n_node = graph.number_of_nodes()

    edge_mask = mask_edge(graph, frac)
    feat = drop_feature(x, feat_drop_rate)

    ng = dgl.graph([])
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]
    ng.add_edges(nsrc, ndst)

    return ng, feat

class rLap():
    def __init__(self, frac, o_v="random", o_n="asc"):
        self.frac = frac
        self.o_v = o_v
        self.o_n = o_n

    def augment(self, graph):

        num_nodes = graph.number_of_nodes()
        self.t = int(self.frac * num_nodes)
        edge_index = graph.edges()
        edge_weights = th.ones((1, edge_index[0].shape[0]))
        edge_info = th.cat((edge_index[0].unsqueeze(0), edge_index[1].unsqueeze(0), edge_weights), dim=0).t()
        ac = ApproximateCholesky()
        ac.setup(edge_info=edge_info.to("cpu"), nrows=num_nodes, ncols=num_nodes, o_v=self.o_v, o_n=self.o_n)

        sparse_edge_info = ac.get_schur_complement(self.t)
        sampled_edge_index = th.Tensor(sparse_edge_info[:,:2]).long().t()
        
        ng = dgl.graph([])
        ng.add_nodes(num_nodes)
        ng.add_edges(sampled_edge_index[0], sampled_edge_index[1])
        del ac
        del sparse_edge_info

        return ng

def rlap_aug(graph, x, feat_drop_rate, frac):

    feat = drop_feature(x, feat_drop_rate)
    rlap_aug = rLap(frac=frac)
    ng = rlap_aug.augment(graph=graph)

    return ng, feat

def drop_feature(x, drop_prob):
    drop_mask = th.empty(
        (x.size(1),),
        dtype=th.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def mask_edge(graph, mask_prob):
    E = graph.number_of_edges()

    mask_rates = th.FloatTensor(np.ones(E) * mask_prob)
    masks = th.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def add_edge(edge_index, ratio):
    num_edges = edge_index.size()[1]
    num_nodes = edge_index.max().item() + 1
    num_add = int(num_edges * ratio)

    new_edge_index = th.randint(0, num_nodes - 1, size=(2, num_add)).to(edge_index.device)
    edge_index = th.cat([edge_index, new_edge_index], dim=1)
    return edge_index

def ea_aug(graph, x, feat_drop_rate, frac):
    n_node = graph.number_of_nodes()

    feat = drop_feature(x, feat_drop_rate)

    ng = dgl.graph([])
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]
    edge_index = th.cat((src.unsqueeze(0), dst.unsqueeze(0)), dim=0)
    new_edge_index = add_edge(edge_index=edge_index, ratio=frac)
    nsrc = new_edge_index[0]
    ndst = new_edge_index[1]
    ng.add_edges(nsrc, ndst)
    return ng, feat

def nd_aug(graph, x, feat_drop_rate, frac):
    n_node = graph.number_of_nodes()
    feat = drop_feature(x, feat_drop_rate)
    ng = dgl.graph([])
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]
    edge_index = th.cat((src.unsqueeze(0), dst.unsqueeze(0)), dim=0)

    new_edge_index, _ = functional.drop_node(edge_index=edge_index, edge_weight=None, keep_prob=frac)

    nsrc = new_edge_index[0]
    ndst = new_edge_index[1]
    ng.add_edges(nsrc, ndst)
    return ng, feat

def markovd_aug(graph, x, feat_drop_rate, frac):
    n_node = graph.number_of_nodes()
    feat = drop_feature(x, feat_drop_rate)
    ng = dgl.graph([])
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]
    edge_index = th.cat((src.unsqueeze(0), dst.unsqueeze(0)), dim=0)

    new_edge_index, _ =  functional.compute_markov_diffusion(
            edge_index, None, alpha=0.2, degree=16, sp_eps=1e-4
    )
    nsrc = new_edge_index[0]
    ndst = new_edge_index[1]
    ng.add_edges(nsrc, ndst)
    return ng, feat

def pprd_aug(graph, x, feat_drop_rate, frac):
    n_node = graph.number_of_nodes()
    feat = drop_feature(x, feat_drop_rate)
    ng = dgl.graph([])
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]
    edge_index = th.cat((src.unsqueeze(0), dst.unsqueeze(0)), dim=0)

    new_edge_index, _ =  functional.compute_ppr(
            edge_index, None, alpha=0.2, eps=1e-4
    )
    nsrc = new_edge_index[0]
    ndst = new_edge_index[1]
    ng.add_edges(nsrc, ndst)
    return ng, feat

def rws_aug(graph, x, feat_drop_rate, frac):
    n_node = graph.number_of_nodes()
    feat = drop_feature(x, feat_drop_rate)
    ng = dgl.graph([])
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]
    edge_index = th.cat((src.unsqueeze(0), dst.unsqueeze(0)), dim=0)

    new_edge_index, _ =  functional.random_walk_subgraph(
        edge_index, None, batch_size=int(frac*n_node))
    nsrc = new_edge_index[0]
    ndst = new_edge_index[1]
    ng.add_edges(nsrc, ndst)
    return ng, feat


def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = th.ones((num_nodes, )).to(edge_index.device).to(th.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x

def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph, tol=1e-05)
    x = [x[i] for i in range(data.num_nodes)]
    return th.tensor(x, dtype=th.float32).to(data.edge_index.device)


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, th.ones_like(edge_weights) * threshold)
    sel_mask = th.bernoulli(1. - edge_weights).to(th.bool)

    return sel_mask


def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(th.float32)
    s_col = th.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(th.float32)
    pv_col = pv[edge_index[1]].to(th.float32)
    s_row = th.log(pv_row)
    s_col = th.log(pv_col)
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
    evc = evc.where(evc > 0, th.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())

def ed_deg_aug(graph, x, feat_drop_rate, frac):
    n_node = graph.number_of_nodes()
    feat = drop_feature(x, feat_drop_rate)
    ng = dgl.graph([])
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]
    edge_index = th.cat((src.unsqueeze(0), dst.unsqueeze(0)), dim=0)

    drop_weights = degree_drop_weights(edge_index=edge_index)
    mask = drop_edge_weighted(
        edge_index=edge_index,
        edge_weights=drop_weights,
        p=frac,
        threshold=0.7
    )
    new_edge_index = edge_index[:, mask]

    nsrc = new_edge_index[0]
    ndst = new_edge_index[1]
    ng.add_edges(nsrc, ndst)
    return ng, feat

def ed_ppr_aug(graph, x, feat_drop_rate, frac):
    n_node = graph.number_of_nodes()
    feat = drop_feature(x, feat_drop_rate)
    ng = dgl.graph([])
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]
    edge_index = th.cat((src.unsqueeze(0), dst.unsqueeze(0)), dim=0)

    drop_weights = pr_drop_weights(edge_index=edge_index)
    mask = drop_edge_weighted(
        edge_index=edge_index,
        edge_weights=drop_weights,
        p=frac,
        threshold=0.7
    )
    new_edge_index = edge_index[:, mask]

    nsrc = new_edge_index[0]
    ndst = new_edge_index[1]
    ng.add_edges(nsrc, ndst)
    return ng, feat

def ed_evc_aug(graph, x, feat_drop_rate, frac):
    n_node = graph.number_of_nodes()
    feat = drop_feature(x, feat_drop_rate)
    ng = dgl.graph([])
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]
    edge_index = th.cat((src.unsqueeze(0), dst.unsqueeze(0)), dim=0)
    device = edge_index.device
    data = Data(x=x, edge_index=edge_index, edge_weights=None).to(device)
    drop_weights = evc_drop_weights(data=data)
    mask = drop_edge_weighted(
        edge_index=edge_index,
        edge_weights=drop_weights,
        p=frac,
        threshold=0.7
    )
    new_edge_index = edge_index[:, mask]

    nsrc = new_edge_index[0]
    ndst = new_edge_index[1]
    ng.add_edges(nsrc, ndst)
    return ng, feat
