import argparse
import torch
import torch.nn.functional as F

torch.cuda.empty_cache()
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch_geometric.transforms as T

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from torch_geometric.utils import subgraph, remove_self_loops
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from augmentor_benchmarks import (
    EdgeAdding,
    EdgeDroppingDegree,
    EdgeDroppingEVC,
    EdgeDroppingPR,
    rLap,
    rLapPPRDiffusion,
)

from sklearn.metrics import f1_score, accuracy_score
from GCL.eval import BaseEvaluator

import numpy as np
from GCL.losses import Loss


class JSD(Loss):
    def __init__(self, discriminator=lambda x, y: x @ y.t()):
        super(JSD, self).__init__()
        self.discriminator = discriminator

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        num_neg = neg_mask.int().sum()
        num_pos = pos_mask.int().sum()
        similarity = self.discriminator(anchor, sample)

        E_pos = (np.log(2) - F.softplus(-similarity * pos_mask)).sum()
        E_pos /= num_pos

        neg_sim = similarity * neg_mask
        E_neg = (F.softplus(-neg_sim) + neg_sim - np.log(2)).sum()
        E_neg /= num_neg

        return E_neg - E_pos


class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class LREvaluator(BaseEvaluator):
    def __init__(
        self,
        num_epochs: int = 2000,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        test_interval: int = 20,
    ):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = Adam(
            classifier.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0
        best_accuracy = 0

        for epoch in range(self.num_epochs):
            classifier.train()
            optimizer.zero_grad()

            output = classifier(x[split["train"]])
            loss = criterion(output_fn(output), y[split["train"]])

            loss.backward()
            optimizer.step()

            if (epoch + 1) % self.test_interval == 0:
                classifier.eval()
                y_test = y[split["test"]].detach().cpu().numpy()
                y_pred = classifier(x[split["test"]]).argmax(-1).detach().cpu().numpy()
                accuracy = accuracy_score(y_test, y_pred)
                test_micro = f1_score(y_test, y_pred, average="micro")
                test_macro = f1_score(y_test, y_pred, average="macro")

                y_val = y[split["valid"]].detach().cpu().numpy()
                y_pred = classifier(x[split["valid"]]).argmax(-1).detach().cpu().numpy()
                val_micro = f1_score(y_val, y_pred, average="micro")

                if val_micro > best_val_micro:
                    best_val_micro = val_micro
                    best_test_micro = test_micro
                    best_test_macro = test_macro
                    best_epoch = epoch
                    best_accuracy = accuracy

        return {
            "micro_f1": best_test_micro,
            "macro_f1": best_test_macro,
            "accuracy": best_accuracy,
        }


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder1, encoder2, augmentor, augmentor_name, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.augmentor = augmentor
        self.augmentor_name = augmentor_name
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        if "diffusion" in self.augmentor_name.lower():
            batch_size = 8192
            clean_edge_index2, clean_edge_weight2 = remove_self_loops(
                edge_index2, edge_weight2
            )
            node_indices = torch.unique(clean_edge_index2)
            num_nodes = node_indices.shape[0]
            print("num nodes: ", num_nodes)
            perm = torch.randperm(num_nodes)
            node_indices = node_indices[perm]
            batch_nodes = node_indices[:batch_size]

            edge_index1, edge_weight1 = subgraph(batch_nodes, edge_index1, edge_weight1)
            edge_index2, edge_weight2 = subgraph(batch_nodes, edge_index2, edge_weight2)
            print(edge_index1.shape, edge_index2.shape)

        z1 = self.encoder1(x1, edge_index1, edge_weight1)
        z2 = self.encoder2(x2, edge_index2, edge_weight2)
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
        g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))
        z1n = self.encoder1(*self.corruption(x1, edge_index1, edge_weight1))
        z2n = self.encoder2(*self.corruption(x2, edge_index2, edge_weight2))
        return z1, z2, g1, g2, z1n, z2n


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z1, z2, g1, g2, z1n, z2n = encoder_model(data.x, data.edge_index)
    loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z1, z2, _, _, _, _ = encoder_model(data.x, data.edge_index)
    z = z1 + z2
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("augmentor", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("num_layers", type=int)
    parser.add_argument("lr", type=float)
    parser.add_argument("wd", type=float)
    parser.add_argument("hidden_dim", type=int)
    parser.add_argument("mode", type=str)
    parser.add_argument("fraction1", type=float)
    parser.add_argument("fraction2", type=float)
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda")
    path = osp.join(osp.expanduser("~"), "datasets")
    datasets = {
        "CORA": lambda: Planetoid(path, name="Cora", transform=T.NormalizeFeatures()),
        "PUBMED": lambda: Planetoid(
            path, name="PubMed", transform=T.NormalizeFeatures()
        ),
        "COAUTHOR-CS": lambda: Coauthor(
            path, name="CS", transform=T.NormalizeFeatures()
        ),
        "COAUTHOR-PHY": lambda: Coauthor(
            path, name="Physics", transform=T.NormalizeFeatures()
        ),
        "AMAZON-PHOTO": lambda: Amazon(
            path, name="Photo", transform=T.NormalizeFeatures()
        ),
    }

    dataset = datasets[args.dataset]()
    data = dataset[0].to(device)
    num_nodes = data.edge_index.max().item() + 1
    fraction1 = args.fraction1
    fraction2 = args.fraction2
    augmentors = {
        "rLap": [
            A.Compose(
                [
                    rLap(frac=fraction1, o_v="random", o_n="asc"),
                    A.FeatureMasking(pf=0.3),
                ]
            ),
            A.Compose(
                [
                    rLap(frac=fraction2, o_v="random", o_n="asc"),
                    A.FeatureMasking(pf=0.3),
                ]
            ),
        ],
        "rLapRandomDesc": [
            A.Compose(
                [
                    rLap(frac=fraction1, o_v="random", o_n="desc"),
                    A.FeatureMasking(pf=0.3),
                ]
            ),
            A.Compose(
                [
                    rLap(frac=fraction2, o_v="random", o_n="desc"),
                    A.FeatureMasking(pf=0.3),
                ]
            ),
        ],
        "rLapRandomRandom": [
            A.Compose(
                [
                    rLap(frac=fraction1, o_v="random", o_n="random"),
                    A.FeatureMasking(pf=0.3),
                ]
            ),
            A.Compose(
                [
                    rLap(frac=fraction2, o_v="random", o_n="random"),
                    A.FeatureMasking(pf=0.3),
                ]
            ),
        ],
        "rLapDegree": [
            A.Compose(
                [
                    rLap(frac=fraction1, o_v="degree", o_n="asc"),
                    A.FeatureMasking(pf=0.3),
                ]
            ),
            A.Compose(
                [
                    rLap(frac=fraction2, o_v="degree", o_n="asc"),
                    A.FeatureMasking(pf=0.3),
                ]
            ),
        ],
        "rLapDegreeDesc": [
            A.Compose(
                [
                    rLap(frac=fraction1, o_v="degree", o_n="desc"),
                    A.FeatureMasking(pf=0.3),
                ]
            ),
            A.Compose(
                [
                    rLap(frac=fraction2, o_v="degree", o_n="desc"),
                    A.FeatureMasking(pf=0.3),
                ]
            ),
        ],
        "rLapDegreeRandom": [
            A.Compose(
                [
                    rLap(frac=fraction1, o_v="degree", o_n="random"),
                    A.FeatureMasking(pf=0.3),
                ]
            ),
            A.Compose(
                [
                    rLap(frac=fraction2, o_v="degree", o_n="random"),
                    A.FeatureMasking(pf=0.3),
                ]
            ),
        ],
        "rLapCoarsen": [
            A.Compose([rLap(frac=fraction1, o_v="coarsen"), A.FeatureMasking(pf=0.3)]),
            A.Compose([rLap(frac=fraction2, o_v="coarsen"), A.FeatureMasking(pf=0.3)]),
        ],
        "EdgeAddition": [
            A.Compose([EdgeAdding(pe=fraction1), A.FeatureMasking(pf=0.3)]),
            A.Compose([EdgeAdding(pe=fraction2), A.FeatureMasking(pf=0.3)]),
        ],
        "EdgeDropping": [
            A.Compose([A.EdgeRemoving(pe=fraction1), A.FeatureMasking(pf=0.3)]),
            A.Compose([A.EdgeRemoving(pe=fraction2), A.FeatureMasking(pf=0.3)]),
        ],
        "EdgeDroppingDegree": [
            A.Compose(
                [
                    EdgeDroppingDegree(p=fraction1, threshold=0.7),
                    A.FeatureMasking(pf=0.3),
                ]
            ),
            A.Compose(
                [
                    EdgeDroppingDegree(p=fraction2, threshold=0.7),
                    A.FeatureMasking(pf=0.3),
                ]
            ),
        ],
        "EdgeDroppingPR": [
            A.Compose(
                [EdgeDroppingPR(p=fraction1, threshold=0.7), A.FeatureMasking(pf=0.3)]
            ),
            A.Compose(
                [EdgeDroppingPR(p=fraction2, threshold=0.7), A.FeatureMasking(pf=0.3)]
            ),
        ],
        "EdgeDroppingEVC": [
            A.Compose(
                [EdgeDroppingEVC(p=fraction1, threshold=0.7), A.FeatureMasking(pf=0.3)]
            ),
            A.Compose(
                [EdgeDroppingEVC(p=fraction2, threshold=0.7), A.FeatureMasking(pf=0.3)]
            ),
        ],
        "NodeDropping": [
            A.Compose([A.NodeDropping(pn=fraction1), A.FeatureMasking(pf=0.3)]),
            A.Compose([A.NodeDropping(pn=fraction2), A.FeatureMasking(pf=0.3)]),
        ],
        "RandomWalkSubgraph": [
            A.Compose([A.Identity(), A.FeatureMasking(pf=0.3)]),
            A.Compose(
                [
                    A.RWSampling(num_seeds=int(fraction2 * num_nodes), walk_length=10),
                    A.FeatureMasking(pf=0.3),
                ]
            ),
        ],
        "PPRDiffusion": [
            A.Compose([A.Identity(), A.FeatureMasking(pf=0.3)]),
            A.Compose(
                [A.PPRDiffusion(alpha=0.2, use_cache=True), A.FeatureMasking(pf=0.3)]
            ),
        ],
        "MarkovDiffusion": [
            A.Compose([A.Identity(), A.FeatureMasking(pf=0.3)]),
            A.Compose(
                [A.MarkovDiffusion(alpha=0.2, use_cache=True), A.FeatureMasking(pf=0.3)]
            ),
        ],
        "rlapPPRDiffusion_8192": [
            A.Compose([A.Identity(), A.FeatureMasking(pf=0.3)]),
            A.Compose(
                [
                    rLapPPRDiffusion(
                        frac=fraction2, o_v="random", o_n="asc", refresh_cache_freq=200
                    ),
                    A.FeatureMasking(pf=0.3),
                ]
            ),
        ],
    }
    aug1, aug2 = augmentors[args.augmentor]

    gconv1 = GConv(
        input_dim=dataset.num_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    gconv2 = GConv(
        input_dim=dataset.num_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    encoder_model = Encoder(
        encoder1=gconv1,
        encoder2=gconv2,
        augmentor=(aug1, aug2),
        augmentor_name=args.augmentor,
        hidden_dim=args.hidden_dim,
    ).to(device)
    contrast_model = DualBranchContrast(loss=JSD(), mode=args.mode).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=args.lr, weight_decay=args.wd)

    early_stopping_tolerance = 50
    current_tolerance = 0
    best_loss = 1e8
    best_epoch = 0
    with tqdm(total=2000, desc="(T)") as pbar:
        for epoch in range(1, 2001):
            loss = train(encoder_model, contrast_model, data, optimizer)
            pbar.set_postfix({"loss": loss})
            pbar.update()
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                current_tolerance = 0
            else:
                current_tolerance += 1

            if current_tolerance == early_stopping_tolerance:
                print("Reached early stopping tolerance!")
                break

    for i in tqdm(range(10)):
        test_result = test(encoder_model, data)
        print(
            f'Test run: {i} : Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}, Acc={test_result["accuracy"]:.4f}'
        )


if __name__ == "__main__":
    main()
