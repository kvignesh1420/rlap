import argparse
import torch
torch.cuda.empty_cache()
import os.path as osp
import GCL.losses as L
from GCL.losses import Loss
import GCL.augmentors as A
import torch.nn.functional as F
from torch import nn
import torch_geometric.transforms as T
import torch_geometric.utils as tg_utils

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split
# from GCL.models import DualBranchContrast
from GCL.models import get_sampler
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from augmentor_benchmarks import EdgeAdding, EdgeDroppingDegree, EdgeDroppingEVC, EdgeDroppingPR, rLap

from sklearn.metrics import f1_score, accuracy_score
from GCL.eval import BaseEvaluator


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

class InfoNCE(Loss):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()

class InfoNCEBatched(Loss):
    def __init__(self, tau, batch_size):
        super(InfoNCEBatched, self).__init__()
        self.tau = tau
        self.batch_size = batch_size

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        device = anchor.device
        num_nodes = anchor.size(0)
        # print("NN: ", num_nodes)
        num_batches = (num_nodes - 1) // self.batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            batch_mask = indices[i*self.batch_size: (i+1)*self.batch_size]
            batch_pos_mask = pos_mask[i*self.batch_size: (i+1)*self.batch_size]
            batch_sim = _similarity(anchor[batch_mask], sample)
            batch_exp_sim = f(batch_sim)
            batch_log_prob = batch_sim - torch.log(batch_exp_sim.sum(dim=1, keepdim=True))
            batch_loss = batch_log_prob * batch_pos_mask
            batch_loss = batch_loss.sum(dim=1)

            losses.append(batch_loss)
            # print(batch_loss.shape)

        losses = torch.cat(losses)
        # print(losses.shape)
        return -losses.mean()

class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
                extra_pos_mask=None, extra_neg_mask=None):
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

        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        return (l1 + l2) * 0.5



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
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
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
        "CORA": lambda: Planetoid(path, name='Cora', transform=T.NormalizeFeatures()),
        "PUBMED": lambda: Planetoid(path, name='PubMed', transform=T.NormalizeFeatures()),
        "COAUTHOR-CS": lambda: Coauthor(path, name="CS", transform=T.NormalizeFeatures()),
        "COAUTHOR-PHY": lambda: Coauthor(path, name="Physics", transform=T.NormalizeFeatures()),
        "AMAZON-PHOTO": lambda: Amazon(path, name='Photo', transform=T.NormalizeFeatures())
    }

    dataset = datasets[args.dataset]()
    data = dataset[0].to(device)
    data.edge_index = tg_utils.to_undirected(data.edge_index)
    num_nodes = data.edge_index.max().item() + 1
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
            A.Compose([A.RWSampling(num_seeds=int(fraction1*num_nodes), walk_length=10), A.FeatureMasking(pf=0.3)]),
            A.Compose([A.RWSampling(num_seeds=int(fraction2*num_nodes), walk_length=10), A.FeatureMasking(pf=0.3)])
        ],
        "PPRDiffusion": [
            A.Compose([A.Identity(), A.FeatureMasking(pf=0.3)]),
            A.Compose([A.PPRDiffusion(alpha=0.2, use_cache=True), A.FeatureMasking(pf=0.3)])
        ],
        "MarkovDiffusion": [
            A.Compose([A.Identity(), A.FeatureMasking(pf=0.3)]),
            A.Compose([A.MarkovDiffusion(alpha=0.2, use_cache=True), A.FeatureMasking(pf=0.3)])
        ],
    }
    aug1, aug2 = augmentors[args.augmentor]

    gconv = GConv(
        input_dim=dataset.num_features,
        hidden_dim=args.hidden_dim,
        activation=torch.nn.PReLU,
        num_layers=args.num_layers).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=args.hidden_dim, proj_dim=args.hidden_dim).to(device)
    contrast_model = DualBranchContrast(loss=InfoNCEBatched(tau=0.4, batch_size=1024), mode=args.mode, intraview_negs=False).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=args.lr, weight_decay=args.wd)

    early_stopping_tolerance = 50
    current_tolerance = 0
    best_loss = 1e8
    best_epoch = 0
    with tqdm(total=2000, desc='(T)') as pbar:
        for epoch in range(1, 2001):
            loss = train(encoder_model, contrast_model, data, optimizer)
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
        test_result = test(encoder_model, data)
        print(f'Test run: {i} : Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}, Acc={test_result["accuracy"]:.4f}')

if __name__ == '__main__':
    main()
