import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

import time
from tqdm import tqdm
from torch.optim import Adam, AdamW
from GCL.eval import get_split, LREvaluator
# from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, Coauthor, WikiCS, SNAPDataset, TUDataset
import numpy as np
from rlap import ApproximateCholesky

from GCL.losses import Loss
from GCL.models import get_sampler


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


class InfoNCE(Loss):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, rw, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        rw += torch.eye(anchor.shape[0])
        sim = sim*rw
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()

class HardnessInfoNCE(Loss):
    def __init__(self, tau, tau_plus=0.1, beta=0.5):
        super(HardnessInfoNCE, self).__init__()
        self.tau = tau
        self.tau_plus = tau_plus
        self.beta = beta

    def compute(self, anchor, sample, pos_mask, neg_mask, rw, *args, **kwargs):
        rw += torch.eye(anchor.shape[0])
        num_neg = neg_mask.int().sum()
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim)

        pos = (exp_sim * pos_mask).sum(dim=1) / pos_mask.int().sum(dim=1)
        imp = torch.exp(rw*rw * (sim * neg_mask))
        reweight_neg = (imp * (exp_sim * neg_mask)).sum(dim=1) / imp.mean(dim=1)
        ng = (-num_neg * self.tau_plus * pos + reweight_neg) / (1 - self.tau_plus)
        ng = torch.clamp(ng, min=num_neg * np.e ** (-1. / self.tau))
        # print(sim.shape, pos.shape, ng.shape)
        log_prob = sim - torch.log(pos + ng)
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return loss.mean()


def add_extra_mask(pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask.bool(), extra_pos_mask.bool()).float()
    if extra_neg_mask is not None:
        neg_mask = torch.bitwise_and(neg_mask.bool(), extra_neg_mask.bool()).float()
    else:
        neg_mask = 1. - pos_mask
    return pos_mask, neg_mask

class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
                extra_pos_mask=None, extra_neg_mask=None, rw1=None, rw2=None):
        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=h1, sample=h2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=h2, sample=h1)
        elif self.mode == 'G2G':
            assert g1 is not None and g2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=g2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=g1)
        else:  # global-to-local
            if batch is None or batch.max().item() + 1 <= 1:  # single graph
                assert all(v is not None for v in [h1, h2, g1, g2, h3, h4])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, neg_sample=h4)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, neg_sample=h3)
            else:  # multiple graphs
                assert all(v is not None for v in [h1, h2, g1, g2, batch])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, batch=batch)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, batch=batch)

        pos_mask1, neg_mask1 = add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        pos_mask2, neg_mask2 = add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        # print("COMPUTING l1")
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, rw=rw1, **self.kwargs)
        # print("COMPUTING l2")
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, rw=rw2, **self.kwargs)

        return (l1 + l2) * 0.5


class rLap(A.Augmentor):
    def __init__(self, t):
        super(rLap, self).__init__()
        self.t = t
        self.ac = ApproximateCholesky()

    def augment(self, g):
        x, edge_index, edge_weights = g.unfold()
        # edge_index, edge_weights = dropout_adj(edge_index, edge_attr=edge_weights, p=self.pe)
        num_nodes = edge_index.max().item() + 1
        # print("EDGE INDEX SHAPE: ", edge_index.shape)
        _edge_weights = edge_weights
        if _edge_weights is None:
            edge_weights = torch.ones((1, edge_index.shape[1])).to(edge_index.device)
        # print("EDGE WT SHAPE: ", edge_weights.shape)
        edge_info = torch.concat((edge_index, edge_weights), dim=0).t()
        # print("EDGE INFO SHAPE: ", edge_info.shape)
        self.ac.setup(edge_info.to("cpu"), num_nodes, num_nodes, "random")
        sparse_edge_info = self.ac.get_schur_complement(self.t)
        # print("sparse_edge_info shape", sparse_edge_info.shape)
        sampled_edge_index = torch.Tensor(sparse_edge_info[:,:2]).long().t().to(edge_index.device)
        # print("Number of edges after aug: ", sampled_edge_index.shape)
        unique_nodes = torch.unique(sampled_edge_index, sorted=False)
        # print("Edges in SC: ", sampled_edge_index)
        # print("Number of unique nodes: ", unique_nodes.shape)
        sampled_edge_weights = torch.Tensor(sparse_edge_info[:,-1]).t().to(edge_index.device)
        # print("Weights: ", sampled_edge_weights)
        # schur complement adjacency matrix
        sc_adj_sparse = torch.sparse_coo_tensor(
            sampled_edge_index,
            sampled_edge_weights,
            (num_nodes, num_nodes)
        ).to(edge_index.device)
        sc_adj = sc_adj_sparse.to_dense()
        # print("sc_adj", sc_adj)
        # validate symmetry in sc_adj
        # print( (sc_adj - sc_adj.t()).norm()/sc_adj.norm() )
        rw_prob = sc_adj/ (sc_adj.sum(dim=1) + 1e-20)
        # print(rw_prob)
        # print("rw_prob_sum: ", rw_prob.sum())

        # if _edge_weights is None:
        #     sampled_edge_weights = None
        # else:
        #     sampled_edge_weights = torch.reshape(sampled_edge_weights, _edge_weights.shape)

        ## mask the features of nodes which are not part of the sub-graph
        # unique_nodes = torch.unique(sampled_edge_index, sorted=False)
        # print("Number of unique nodes: ", unique_nodes.shape)
        # x_new = x.detach().clone()
        # for i in range(num_nodes):
        #     if i not in unique_nodes:
        #         x_new[i,:] = 0.0
        return A.Graph(x=x, edge_index=sampled_edge_index, edge_weights=sampled_edge_weights)



class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.shape[0]
        aug1, aug2 = self.augmentor
        s = time.time()
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        sc_adj_sparse = torch.sparse_coo_tensor(
            edge_index1,
            edge_weight1,
            (num_nodes, num_nodes)
        ).to(edge_index.device)
        sc_adj = sc_adj_sparse.to_dense()
        # print("sc_adj", sc_adj)
        # validate symmetry in sc_adj
        # print( (sc_adj - sc_adj.t()).norm()/sc_adj.norm() )
        rw1 = sc_adj/ (sc_adj.sum(dim=1) + 1e-20)
        e1 = time.time()
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        sc_adj_sparse = torch.sparse_coo_tensor(
            edge_index2,
            edge_weight2,
            (num_nodes, num_nodes)
        ).to(edge_index.device)
        sc_adj = sc_adj_sparse.to_dense()
        # print("sc_adj", sc_adj)
        # validate symmetry in sc_adj
        # print( (sc_adj - sc_adj.t()).norm()/sc_adj.norm() )
        rw2 = sc_adj/ (sc_adj.sum(dim=1) + 1e-20)
        e2 = time.time()
        # print("LATENCIES OF AUG1: {}, AUG2: {}".format(e1-s, e2-e1))
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        # print("Sizes of z, z1, z2: {} {} {} ".format(z.shape, z1.shape, z2.shape))
        # print("Sizes of ei, ei1, ei2", edge_index.shape, edge_index1.shape, edge_index2.shape)
        # # print("z, z1, z2: {} {} {} ".format(z, z1, z2))
        # x_sum = x.sum(dim=0)
        # x1_sum = x1.sum(dim=0)
        # x2_sum = x2.sum(dim=0)
        # print(torch.nonzero(x_sum).shape, torch.nonzero(x1_sum).shape, torch.nonzero(x2_sum).shape)

        # z_sum = z.sum(dim=0)
        # z1_sum = z1.sum(dim=0)
        # z2_sum = z2.sum(dim=0)
        # print(torch.nonzero(z_sum).shape, torch.nonzero(z1_sum).shape, torch.nonzero(z2_sum).shape)

        return z, z1, z2, rw1, rw2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2, rw1, rw2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1=h1, h2=h2, rw1=rw1, rw2=rw2)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    device = torch.device('cpu')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
    # dataset = Coauthor(path, name="CS", transform=T.NormalizeFeatures())
    # dataset = WikiCS(path, transform=T.NormalizeFeatures())
    # dataset = SNAPDataset(path, name="soc-Pokec", transform=T.NormalizeFeatures())
    # dataset = SNAPDataset(path, name="ego-facebook", transform=T.NormalizeFeatures())
    # dataset = TUDataset(path, name="REDDIT-BINARY", transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    # row, col = data.edge_index
    # row = torch.cat([row, col])
    # col = torch.cat([col, row])
    # data.edge_index = torch.stack([row, col], dim=0)
    # data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)
    print(" IS DATA UNDIRECTED: ", data.is_undirected())

    # A.PPRDiffusion(alpha=0.2, use_cache=False)
    aug1 = A.Compose([rLap(1500), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([rLap(1500), A.FeatureMasking(pf=0.3)])

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=128, activation=torch.nn.ReLU, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=128, proj_dim=128).to(device)
    contrast_model = DualBranchContrast(loss=HardnessInfoNCE(tau=0.2), mode='L2L', intraview_negs=False).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.005, weight_decay=1e-5)

    with tqdm(total=1000, desc='(T)') as pbar:
        for epoch in range(1, 1001):
            loss = train(encoder_model, contrast_model, data, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()
            if epoch%50==0:
                test_result = test(encoder_model, data)
                print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
