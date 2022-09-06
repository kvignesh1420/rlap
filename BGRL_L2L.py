import copy
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import BootstrapContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import WikiCS, Planetoid, Coauthor
import numpy as np
from rlap import ApproximateCholesky

class rLAP(A.Augmentor):
    def __init__(self, t):
        super(rLAP, self).__init__()
        self.t = t
        self.ac = ApproximateCholesky()

    def augment(self, g):
        # print(" in augment ")
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
        self.ac.setup(edge_info.to("cpu"), num_nodes, num_nodes, "degree")
        sparse_edge_info = self.ac.get_schur_complement(self.t)
        # print("sparse_edge_info shape", sparse_edge_info.shape)
        sampled_edge_index = torch.Tensor(sparse_edge_info[:,:2]).long().t().to(edge_index.device)
        sampled_edge_weights = torch.Tensor(sparse_edge_info[:,-1]).t().to(edge_index.device)
        if _edge_weights is None:
            sampled_edge_weights = None
        else:
            sampled_edge_weights = torch.reshape(sampled_edge_weights, _edge_weights.shape)
        # print(" done augment ")
        return A.Graph(x=x, edge_index=sampled_edge_index, edge_weights=sampled_edge_weights)



class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch'):
        super(Encoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight1)
        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight2)

        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    _, _, h1_pred, h2_pred, h1_target, h2_target = encoder_model(data.x, data.edge_index, data.edge_attr)
    loss = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred, h1_target=h1_target.detach(), h2_target=h2_target.detach())
    loss.backward()
    optimizer.step()
    encoder_model.update_target_encoder(0.99)
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    h1, h2, _, _, _, _ = encoder_model(data.x, data.edge_index)
    z = torch.cat([h1, h2], dim=1)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    device = torch.device('cuda')
    # path = osp.join(osp.expanduser('~'), 'datasets', 'WikiCS')
    # dataset = WikiCS(path, transform=T.NormalizeFeatures())
    path = osp.join(osp.expanduser('~'), 'datasets')
    # dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
    dataset = Coauthor(path, name="CS", transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    aug1 = A.Compose([rLAP(15000), A.PPRDiffusion(alpha=0.2, use_cache=False),A.PPRDiffusion(alpha=0.2, use_cache=False), A.FeatureMasking(pf=0.1)])
    aug2 = A.Compose([rLAP(15000), A.FeatureMasking(pf=0.1)])

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=256, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=256).to(device)
    contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode='L2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, 1001):
            loss = train(encoder_model, contrast_model, data, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()
            if epoch%50 == 0:
                test_result = test(encoder_model, data)
                print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
