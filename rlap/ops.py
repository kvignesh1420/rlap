import torch
from torch import Tensor
from . import _C


def approximate_cholesky(
    edge_info: Tensor, num_nodes: int, num_remove: int, o_v: str, o_n: str
) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.extension_cpp.approximate_cholesky.default(
        edge_info=edge_info,
        num_nodes=num_nodes,
        num_remove=num_remove,
        o_v=o_v,
        o_n=o_n,
    )


def identity(a: Tensor) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.extension_cpp.identity.default(a=a)
