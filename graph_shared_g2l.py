import argparse
import torch
import torch.nn.functional as F
torch.cuda.empty_cache()
import copy
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split
from GCL.models import BootstrapContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from augmentor_benchmarks import EdgeAdding, EdgeDroppingDegree, EdgeDroppingEVC, EdgeDroppingPR, rLap

from sklearn.metrics import f1_score, accuracy_score
from GCL.eval import BaseEvaluator


class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 2000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 20):
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
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
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

            output = classifier(x[split['train']])
            loss = criterion(output_fn(output), y[split['train']])

            loss.backward()
            optimizer.step()

            if (epoch + 1) % self.test_interval == 0:
                classifier.eval()
                y_test = y[split['test']].detach().cpu().numpy()
                y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                accuracy = accuracy_score(y_test, y_pred)
                test_micro = f1_score(y_test, y_pred, average='micro')
                test_macro = f1_score(y_test, y_pred, average='macro')

                y_val = y[split['valid']].detach().cpu().numpy()
                y_pred = classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                val_micro = f1_score(y_val, y_pred, average='micro')

                if val_micro > best_val_micro:
                    best_val_micro = val_micro
                    best_test_micro = test_micro
                    best_test_macro = test_macro
                    best_epoch = epoch
                    best_accuracy = accuracy

        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro,
            'accuracy': best_accuracy
        }




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


def make_gin_conv(input_dim: int, out_dim: int) -> GINConv:
    mlp = torch.nn.Sequential(
        torch.nn.Linear(input_dim, out_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(out_dim, out_dim))
    return GINConv(mlp)


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(make_gin_conv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(make_gin_conv(hidden_dim, hidden_dim))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index)
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

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight1)
        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight2)

        g1 = global_add_pool(h1, batch)
        h1_pred = self.predictor(h1_online)
        g2 = global_add_pool(h2, batch)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)
            g1_target = global_add_pool(h1_target, batch)
            g2_target = global_add_pool(h2_target, batch)

        return g1, g2, h1_pred, h2_pred, g1_target, g2_target


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    total_loss = 0

    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32).to(data.batch.device)

        optimizer.zero_grad()
        _, _, h1_pred, h2_pred, g1_target, g2_target = encoder_model(data.x, data.edge_index, batch=data.batch)

        loss = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred,
                              g1_target=g1_target.detach(), g2_target=g2_target.detach(), batch=data.batch)
        loss.backward()
        optimizer.step()
        encoder_model.update_target_encoder(0.99)

        total_loss += loss.item()

    return total_loss


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        g1, g2, _, _, _, _ = encoder_model(data.x, data.edge_index, batch=data.batch)
        z = torch.cat([g1, g2], dim=1)
        x.append(z)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = LREvaluator()(x, y, split)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('augmentor', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('num_layers', type=int)
    parser.add_argument('lr', type=float)
    parser.add_argument('wd', type=float)
    parser.add_argument('hidden_dim', type=int)
    parser.add_argument('mode', type=str)
    parser.add_argument('fraction1', type=float)
    parser.add_argument('fraction2', type=float)
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    datasets = {
        "PROTEINS": lambda: TUDataset(path, name='PROTEINS_full'),
        "IMDB-BINARY": lambda: TUDataset(path, name='IMDB-BINARY'),
        "IMDB-MULTI": lambda: TUDataset(path, name='IMDB-MULTI'),
        "MUTAG": lambda: TUDataset(path, name='MUTAG'),
        "NCI1": lambda: TUDataset(path, name='NCI1'),
    }

    dataset = datasets[args.dataset]()
    dataloader = DataLoader(dataset, batch_size=128)
    input_dim = max(dataset.num_features, 1)

    fraction1 = args.fraction1
    fraction2 = args.fraction2
    augmentors = {
        "rLap": [
            A.Compose([rLap(frac=fraction1, o_v="random", o_n="asc"), A.FeatureMasking(pf=0.3)]),
            A.Compose([rLap(frac=fraction2, o_v="random", o_n="asc"), A.FeatureMasking(pf=0.3)])
        ],
        "rLapRandomDesc": [
            A.Compose([rLap(frac=fraction1, o_v="random", o_n="desc"), A.FeatureMasking(pf=0.3)]),
            A.Compose([rLap(frac=fraction2, o_v="random", o_n="desc"), A.FeatureMasking(pf=0.3)])
        ],
        "rLapRandomRandom": [
            A.Compose([rLap(frac=fraction1, o_v="random", o_n="random"), A.FeatureMasking(pf=0.3)]),
            A.Compose([rLap(frac=fraction2, o_v="random", o_n="random"), A.FeatureMasking(pf=0.3)])
        ],
        "rLapDegree": [
            A.Compose([rLap(frac=fraction1, o_v="degree", o_n="asc"), A.FeatureMasking(pf=0.3)]),
            A.Compose([rLap(frac=fraction2, o_v="degree", o_n="asc"), A.FeatureMasking(pf=0.3)])
        ],
        "rLapDegreeDesc": [
            A.Compose([rLap(frac=fraction1, o_v="degree", o_n="desc"), A.FeatureMasking(pf=0.3)]),
            A.Compose([rLap(frac=fraction2, o_v="degree", o_n="desc"), A.FeatureMasking(pf=0.3)])
        ],
        "rLapDegreeRandom": [
            A.Compose([rLap(frac=fraction1, o_v="degree", o_n="random"), A.FeatureMasking(pf=0.3)]),
            A.Compose([rLap(frac=fraction2, o_v="degree", o_n="random"), A.FeatureMasking(pf=0.3)])
        ],
        "rLapCoarsen": [
            A.Compose([rLap(frac=fraction1, o_v="coarsen"), A.FeatureMasking(pf=0.3)]),
            A.Compose([rLap(frac=fraction2, o_v="coarsen"), A.FeatureMasking(pf=0.3)])
        ],
        "EdgeAddition": [
            A.Compose([EdgeAdding(pe=fraction1), A.FeatureMasking(pf=0.3)]),
            A.Compose([EdgeAdding(pe=fraction2), A.FeatureMasking(pf=0.3)])
        ],
        "EdgeDropping": [
            A.Compose([A.EdgeRemoving(pe=fraction1), A.FeatureMasking(pf=0.3)]),
            A.Compose([A.EdgeRemoving(pe=fraction2), A.FeatureMasking(pf=0.3)])
        ],
        "EdgeDroppingDegree": [
            A.Compose([EdgeDroppingDegree(p=fraction1, threshold=0.7), A.FeatureMasking(pf=0.3)]),
            A.Compose([EdgeDroppingDegree(p=fraction2, threshold=0.7), A.FeatureMasking(pf=0.3)])
        ],
        "EdgeDroppingPR": [
            A.Compose([EdgeDroppingPR(p=fraction1, threshold=0.7), A.FeatureMasking(pf=0.3)]),
            A.Compose([EdgeDroppingPR(p=fraction2, threshold=0.7), A.FeatureMasking(pf=0.3)])
        ],
        "EdgeDroppingEVC": [
            A.Compose([EdgeDroppingEVC(p=fraction1, threshold=0.7), A.FeatureMasking(pf=0.3)]),
            A.Compose([EdgeDroppingEVC(p=fraction2, threshold=0.7), A.FeatureMasking(pf=0.3)])
        ],
        "NodeDropping": [
            A.Compose([A.NodeDropping(pn=fraction1), A.FeatureMasking(pf=0.3)]),
            A.Compose([A.NodeDropping(pn=fraction2), A.FeatureMasking(pf=0.3)])
        ],
        "RandomWalkSubgraph": [
            A.Compose([A.RWSampling(num_seeds=1000, walk_length=10), A.FeatureMasking(pf=0.3)]),
            A.Compose([A.RWSampling(num_seeds=1000, walk_length=10), A.FeatureMasking(pf=0.3)])
        ],
        "PPRDiffusion": [
            A.Compose([A.Identity(), A.FeatureMasking(pf=0.3)]),
            A.Compose([A.PPRDiffusion(alpha=0.2, use_cache=False), A.FeatureMasking(pf=0.3)])
        ],
        "MarkovDiffusion": [
            A.Compose([A.Identity(), A.FeatureMasking(pf=0.3)]),
            A.Compose([A.MarkovDiffusion(alpha=0.2, use_cache=False), A.FeatureMasking(pf=0.3)])
        ],
    }
    aug1, aug2 = augmentors[args.augmentor]

    gconv = GConv(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=args.hidden_dim).to(device)
    contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode=args.mode).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=args.lr, weight_decay=args.wd)

    early_stopping_tolerance = 50
    current_tolerance = 0
    best_loss = 1e8
    best_epoch = 0
    with tqdm(total=2000, desc='(T)') as pbar:
        for epoch in range(1, 2001):
            loss = train(encoder_model, contrast_model, dataloader, optimizer)
            pbar.set_postfix({'loss': loss})
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
        test_result = test(encoder_model, dataloader)
        print(f'Test run: {i} : Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}, Acc={test_result["accuracy"]:.4f}')


if __name__ == '__main__':
    main()
