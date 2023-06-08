import os.path as osp
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
import torch_geometric.transforms as T
import torch_geometric.utils as tg_utils
from rlap.python.api import ApproximateCholesky
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def get_rlap_sc_stats(data, batch_count, nodes_to_eliminate, o_v, o_n):
    edge_index = data.edge_index
    try:
        data.__getattribute__("edge_weights")
    except AttributeError:
        edge_weights = None
    _edge_weights = edge_weights
    if _edge_weights is None:
        edge_weights = torch.ones((1, edge_index.shape[1])).to(edge_index.device)

    num_unique_nodes = []
    num_edges = []
    max_sv = []
    num_nodes = edge_index.max().item() + 1
    ac = ApproximateCholesky()
    for batch in tqdm(range(batch_count)):
        edge_info = torch.concat((edge_index, edge_weights), dim=0).t()
        ac.setup(edge_info.to("cpu"), num_nodes, num_nodes, o_v, o_n)
        sparse_edge_info = ac.get_schur_complement(nodes_to_eliminate)
        edge_index = torch.Tensor(sparse_edge_info[:,:2]).long().t().to(edge_index.device)
        edge_weights = torch.Tensor(sparse_edge_info[:,-1]).t().to(edge_index.device).unsqueeze(0)
        num_unique_nodes.append(torch.unique(edge_index).shape)
        unique_nodes = torch.unique(edge_index)
        num_nodes = unique_nodes.shape[0]
        edge_weights = torch.Tensor(sparse_edge_info[:,-1]).t()
        edge_index, edge_weights = tg_utils.subgraph(unique_nodes, edge_index=edge_index, edge_attr=edge_weights, relabel_nodes=True)
        edge_weights = edge_weights.unsqueeze(0)
        num_edges.append(edge_index.shape[1])

        adj = tg_utils.to_dense_adj(edge_index=edge_index)[0]
        U, S, V = torch.svd_lowrank(adj)
        max_sv.append(S[0])
    del ac
    return max_sv, num_unique_nodes, num_edges


def plot_sv_trend(dataset_name, trend):
    strategies = sorted(list(trend.keys()))
    for strategy, color in zip(strategies, colors):
        max_sv = torch.Tensor(trend[strategy]["max_sv"]).squeeze()
        max_sv_mean = torch.mean(max_sv, axis=0)
        max_sv_std = torch.std(max_sv, axis=0)
        x = torch.arange(1, max_sv_mean.shape[0]+1)/(batch_count/frac)
        strategy_name = "rLap:{}".format(strategy) if "coarsen" not in strategy else strategy.split("_")[0]
        plt.plot(x, max_sv_mean, color=color, label="{}".format(strategy_name))
        plt.fill_between(
            x, 
            max_sv_mean-max_sv_std,
            max_sv_mean+max_sv_std,
            color=color,
            alpha=0.2,
            interpolate=True,
        )
        plt.xlabel("Fraction of perturbation")
        plt.ylabel("Maximum singular value")
        plt.legend()
    plt.tight_layout()
    plt.savefig("plots/{}_max_sv_trend.png".format(dataset_name))
    plt.clf()


def plot_edge_count_trend(dataset_name, trend):
    strategies = sorted(list(trend.keys()))
    for strategy, color in zip(strategies, colors):
        edge_count = torch.Tensor(trend[strategy]["edge_count"]).squeeze()
        edge_count_mean = torch.mean(edge_count, axis=0)
        edge_count_std = torch.std(edge_count, axis=0)
        x = torch.arange(1, edge_count_mean.shape[0]+1)/(batch_count/frac)
        strategy_name = "rLap:{}".format(strategy) if "coarsen" not in strategy else strategy.split("_")[0]
        plt.plot(x, edge_count_mean, color=color, label="{}".format(strategy_name))
        plt.fill_between(
            x, 
            edge_count_mean-edge_count_std,
            edge_count_mean+edge_count_std,
            color=color,
            alpha=0.2,
            interpolate=True,
        )
        plt.xlabel("Fraction of perturbation")
        plt.ylabel("Edge count")
        plt.legend()
    plt.tight_layout()
    plt.savefig("plots/{}_edge_count_trend.png".format(dataset_name))
    plt.clf()


if __name__ == "__main__":

    Path("plots").mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')
    path = osp.join(osp.expanduser('~'), 'datasets')
    datasets = {
        "CORA": lambda: Planetoid(path, name='Cora', transform=T.NormalizeFeatures()),
        "AMAZON-PHOTO": lambda: Amazon(path, name='Photo', transform=T.NormalizeFeatures()),
        "PUBMED": lambda: Planetoid(path, name='PubMed', transform=T.NormalizeFeatures()),
        "COAUTHOR-CS": lambda: Coauthor(path, name="CS", transform=T.NormalizeFeatures()),
        "COAUTHOR-PHY": lambda: Coauthor(path, name="Physics", transform=T.NormalizeFeatures()),
    }

    o_v_list = ["random"]
    # o_v_list = ["random", "degree", "coarsen"]
    o_n_list = ["asc", "desc", "random"]
    colors = ["red", "blue", "green"]
    num_runs = 10
    frac = 0.5
    batch_count = 10
    batch_frac = frac/batch_count

    for dataset_name in datasets.keys():
        sc_trend = {}
        for o_v in o_v_list:
            for o_n in o_n_list:
                sc_trend["{}_{}".format(o_v, o_n)] = defaultdict(list)

        for o_v in o_v_list:
            for o_n in o_n_list:
                print("DATASET: {} o_v: {} o_n: {}".format(dataset_name, o_v, o_n))
                for _ in range(num_runs):
                    dataset = datasets[dataset_name]()
                    data = dataset[0].to(device)
                    data.edge_index = tg_utils.to_undirected(data.edge_index)
                    num_nodes = data.edge_index.max().item() + 1
                    batch_nodes_to_eliminate = int(batch_frac*num_nodes)
                    data = get_rlap_sc_stats(data, batch_count, batch_nodes_to_eliminate, o_v, o_n)
                    sc_trend["{}_{}".format(o_v, o_n)]["max_sv"].append(data[0])
                    sc_trend["{}_{}".format(o_v, o_n)]["node_count"].append(data[1])
                    sc_trend["{}_{}".format(o_v, o_n)]["edge_count"].append(data[2])

        plot_sv_trend(dataset_name, sc_trend)
        plot_edge_count_trend(dataset_name, sc_trend)

