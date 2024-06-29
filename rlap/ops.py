from typing import Optional
import torch
from torch import Tensor
from . import _C


def approximate_cholesky(
    edge_index: Tensor,
    edge_weights: Optional[Tensor],
    num_nodes: int,
    num_remove: int,
    o_v: str,
    o_n: str,
) -> Tensor:
    """
    Compute the randomized Schur complement of the graph Laplacian.

    Parameters:
    -----------
    edge_index : Tensor
        A 2d tensor representing the edges in a graph.
    edge_weights: Tensor
        A 1d tensor representing the weights of the edges.
    num_nodes : int
        The total number of nodes in the graph.
    num_remove : int
        The number of nodes to remove for the Schur complement computation.
    o_v : str
        The strategy to use for eliminating the nodes.
        Choose from ["random", "degree", "coarsen"].
    o_n : str
        The strategy to use for eliminating the neighbors of a node
        (based on the sum of weights of edges that connect to the neighbors).
        Choose from ["asc", "desc", "random"].

    Returns:
    --------
    Tensor
        A tensor representing the concatenation of edge_index, edge_weight
        of the graph corresponding to the randomized schur complement.
    """

    assert edge_index.shape[0] == 2

    if edge_weights is None:
        edge_weights = torch.ones((1, edge_index.shape[1])).to(edge_index.device)
    edge_info = torch.concat((edge_index, edge_weights), dim=0).t().double().cpu()

    assert o_v in ["random", "degree", "coarsen"]
    assert o_n in ["asc", "desc", "random"]

    return torch.ops.extension_cpp.approximate_cholesky.default(
        edge_info=edge_info,
        num_nodes=num_nodes,
        num_remove=num_remove,
        o_v=o_v,
        o_n=o_n,
    )


def identity(a: Tensor) -> Tensor:
    """Verification op for Torch -> Eigen -> Torch conversion."""
    return torch.ops.extension_cpp.identity.default(a=a)
