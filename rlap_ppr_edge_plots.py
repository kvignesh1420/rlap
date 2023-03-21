"""
Benchmark the resource consumption of graph augmentors.
"""
from collections import defaultdict
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import compute_ppr
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import subgraph, remove_self_loops
from torch_geometric.datasets import Coauthor
import numpy as np
from augmentor_benchmarks import rLapPPRDiffusion

import matplotlib.pyplot as plt
from pathlib import Path


if __name__ == "__main__":

    Path("plots").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    path = osp.join(osp.expanduser('~'), 'datasets')
    datasets = {
        "COAUTHOR-CS": lambda: Coauthor(path, name="CS", transform=T.NormalizeFeatures()),
        "COAUTHOR-PHY": lambda: Coauthor(path, name="Physics", transform=T.NormalizeFeatures()),
    }

    for fraction in [0.1, 0.2, 0.3, 0.4, 0.5]:
        rlap_ppr_aug = A.Compose([rLapPPRDiffusion(frac=fraction, o_v="random", o_n="asc", refresh_cache_freq=200, use_cache=False)])
        ppr_aug = A.Compose([A.PPRDiffusion(alpha=0.2, use_cache=False)])
        edge_counts = defaultdict(list)
        fig, ax = plt.subplots()
        for d_name in ["COAUTHOR-CS", "COAUTHOR-PHY"]:
            for aug_name, aug in {"PPR": ppr_aug, "rLapPPR": rlap_ppr_aug}.items():
                print(d_name, aug_name)
                dataset = datasets[d_name]()
                data = dataset[0].to(device)
                x, edge_index, edge_weight = aug(data.x, data.edge_index, data.edge_weight)

                batch_size = 8192
                clean_edge_index, clean_edge_weight = remove_self_loops(edge_index, edge_weight)
                node_indices = torch.unique(clean_edge_index)
                num_nodes = node_indices.shape[0]
                print("num nodes: ", num_nodes)
                perm = torch.randperm(num_nodes)
                node_indices = node_indices[perm]
                batch_nodes = node_indices[:batch_size]
                edge_index, edge_weight = subgraph(batch_nodes, edge_index, edge_weight)
                print(edge_index.shape)
                N = torch.unique(edge_index).size()
                print(edge_index.shape, N)
                label = "{}:{}".format(d_name, aug_name)
                edge_counts[label].append(edge_index.shape[1])


        techniques = []
        means = []
        stds = []
        for technique, counts in edge_counts.items():
            techniques.append(technique)
            means.append(np.round(np.mean(counts), 3))
            stds.append(np.round(np.std(counts), 3))

        ind = np.arange(len(techniques))
        fig, ax = plt.subplots()
        ax.bar(ind, means, align='center')
        ax.set_ylabel('Edge count in sub-graph')
        ax.set_xticks(ind)
        ax.set_xticklabels(techniques, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('plots/ppr_rlapppr_edge_counts_frac_{}.png'.format(fraction))
        plt.show()
        plt.clf()
