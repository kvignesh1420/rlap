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
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, Coauthor, WikiCS, SNAPDataset, TUDataset
import numpy as np
from rlap import ApproximateCholesky

class rLAP(A.Augmentor):
    def __init__(self, t):
        super(rLAP, self).__init__()
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
        sampled_edge_weights = torch.Tensor(sparse_edge_info[:,-1]).t().to(edge_index.device)
        if _edge_weights is None:
            sampled_edge_weights = None
        else:
            sampled_edge_weights = torch.reshape(sampled_edge_weights, _edge_weights.shape)
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
        aug1, aug2 = self.augmentor
        s = time.time()
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        e1 = time.time()
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        e2 = time.time()
        # print("LATENCIES OF AUG1: {}, AUG2: {}".format(e1-s, e2-e1))
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
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
    aug1 = A.Compose([rLAP(1500), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([rLAP(1500), A.FeatureMasking(pf=0.3)])

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=128, activation=torch.nn.ReLU, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=128, proj_dim=128).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)

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
