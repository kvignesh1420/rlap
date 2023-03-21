import torch as th
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


def ea_aug(graph, x, feat_drop_rate, frac):
    n_node = graph.number_of_nodes()

    feat = drop_feature(x, feat_drop_rate)

    ng = dgl.graph([])
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]
    edge_index = th.cat((src.unsqueeze(0), dst.unsqueeze(0)), dim=0)
    new_edge_index = functional.add_edge(edge_index=edge_index, ratio=frac)
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
