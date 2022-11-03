import argparse
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
from torch import nn
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, Coauthor, WikiCS
from rlap import ApproximateCholesky
from augmentor_benchmarks import EdgeAdding, EdgeDroppingDegree, EdgeDroppingEVC, EdgeDroppingPR

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


class rLap(A.Augmentor):
    def __init__(self, t):
        super(rLap, self).__init__()
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
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    datasets = {
        "CORA": lambda: Planetoid(path, name='Cora', transform=T.NormalizeFeatures()),
        "WIKI-CS": lambda: WikiCS(path, transform=T.NormalizeFeatures()),
        "COAUTHOR-CS": lambda: Coauthor(path, name="CS", transform=T.NormalizeFeatures()),
        "COAUTHOR-PHY": lambda: Coauthor(path, name="Physics", transform=T.NormalizeFeatures()),
    }

    dataset = datasets[args.dataset]()
    data = dataset[0].to(device)
    num_nodes = data.edge_index.max().item() + 1
    fraction = 0.3
    augmentors = {
        "rLap": [
            A.Compose([rLap(int(fraction*num_nodes)), A.FeatureMasking(pf=0.3)]),
            A.Compose([rLap(int(fraction*num_nodes)), A.FeatureMasking(pf=0.3)])
        ],
        "EdgeAddition": [
            A.Compose([EdgeAdding(pe=fraction), A.FeatureMasking(pf=0.3)]),
            A.Compose([EdgeAdding(pe=fraction), A.FeatureMasking(pf=0.3)])
        ],
        "EdgeDropping": [
            A.Compose([A.EdgeRemoving(pe=fraction), A.FeatureMasking(pf=0.3)]),
            A.Compose([A.EdgeRemoving(pe=fraction), A.FeatureMasking(pf=0.3)])
        ],
        "EdgeDroppingDegree": [
            A.Compose([EdgeDroppingDegree(p=fraction, threshold=0.7), A.FeatureMasking(pf=0.3)]),
            A.Compose([EdgeDroppingDegree(p=fraction, threshold=0.7), A.FeatureMasking(pf=0.3)])
        ],
        "EdgeDroppingPR": [
            A.Compose([EdgeDroppingPR(p=fraction, threshold=0.7), A.FeatureMasking(pf=0.3)]),
            A.Compose([EdgeDroppingPR(p=fraction, threshold=0.7), A.FeatureMasking(pf=0.3)])
        ],
        "EdgeDroppingEVC": [
            A.Compose([EdgeDroppingEVC(p=fraction, threshold=0.7), A.FeatureMasking(pf=0.3)]),
            A.Compose([EdgeDroppingEVC(p=fraction, threshold=0.7), A.FeatureMasking(pf=0.3)])
        ],
        "NodeDropping": [
            A.Compose([A.NodeDropping(pn=fraction), A.FeatureMasking(pf=0.3)]),
            A.Compose([A.NodeDropping(pn=fraction), A.FeatureMasking(pf=0.3)])
        ],
        "RandomWalkSubgraph": [
            A.Compose([A.RWSampling(num_seeds=int(fraction*num_nodes), walk_length=10), A.FeatureMasking(pf=0.3)]),
            A.Compose([A.RWSampling(num_seeds=int(fraction*num_nodes), walk_length=10), A.FeatureMasking(pf=0.3)])
        ],
        "PPRDiffusion": [
            A.Compose([A.Identity(), A.FeatureMasking(pf=0.3)]),
            A.Compose([A.PPRDiffusion(alpha=0.2, use_cache=True), A.FeatureMasking(pf=0.3)])
        ],
        "MarkovDiffusion": [
            A.Compose([A.Identity(), A.FeatureMasking(pf=0.3)]),
            A.Compose([A.MarkovDiffusion(alpha=0.2, use_cache=True), A.FeatureMasking(pf=0.3)])
        ],
        # For this combination, leverage the fact that we need same set of nodes across views.
        # We can thus, significantly compress the size of 'x' being passed to PPR.
        # "rLap+PPRDiffusion": A.Compose([rLap(int(fraction*num_nodes)), A.PPRDiffusion(alpha=0.2, use_cache=False)]),
        # "rLap+MarkovDiffusion": A.Compose([rLap(int(fraction*num_nodes)), A.MarkovDiffusion(alpha=0.2, use_cache=False)]),
    }
    aug1, aug2 = augmentors[args.augmentor]

    gconv = GConv(
        input_dim=dataset.num_features,
        hidden_dim=args.hidden_dim,
        activation=torch.nn.ReLU,
        num_layers=args.num_layers).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=args.hidden_dim, proj_dim=args.hidden_dim).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode=args.mode, intraview_negs=False).to(device)

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

            if optimizer.param_groups[0]['lr'] < 1e-5:
                print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
                break

    for i in tqdm(range(10)):
        test_result = test(encoder_model, data)
        print(f'Test run: {i} : Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}, Acc={test_result["accuracy"]:.4f}')

if __name__ == '__main__':
    main()
