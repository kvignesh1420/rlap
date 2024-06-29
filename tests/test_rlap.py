import torch
from torch_geometric.utils import (
    barabasi_albert_graph,
    to_undirected,
    to_scipy_sparse_matrix,
)
import unittest
from torch.testing._internal.common_utils import TestCase
import rlap


class TestIdentity(TestCase):
    def _test_fn(self):
        a = torch.randn(100, 100).double()
        rlap_a = rlap.ops.identity(a=a)
        assert torch.allclose(a, rlap_a, atol=1e-8)

    def test_fn(self):
        for _ in range(10):
            self._test_fn()


class TestApproximateCholesky(TestCase):

    def create_dummy_graph(self, num_nodes=100):
        # Create a random graph using Barabasi-Albert model (preferential attachment)
        edge_index = barabasi_albert_graph(
            num_nodes=num_nodes, num_edges=num_nodes // 2
        )
        edge_index = to_undirected(edge_index=edge_index, num_nodes=num_nodes)
        return edge_index

    def _is_symmetric(self, edge_index: torch.Tensor, num_nodes: int):
        adj_sparse = to_scipy_sparse_matrix(edge_index=edge_index, num_nodes=num_nodes)
        adj_torch = torch.tensor(adj_sparse.todense())
        assert torch.allclose(adj_torch, adj_torch.t(), atol=1e-8)

    def _test_fn(self):
        num_nodes = 100
        edge_index = self.create_dummy_graph(num_nodes=num_nodes)
        self._is_symmetric(edge_index=edge_index, num_nodes=num_nodes)
        # print(f"edge_index shape: {edge_index.shape}")
        edge_weights = torch.ones((1, edge_index.shape[1])).to(edge_index.device)
        num_remove = 50
        o_v = "random"
        o_n = "asc"

        sc_edge_info = rlap.ops.approximate_cholesky(
            edge_index=edge_index,
            edge_weights=edge_weights,
            num_nodes=num_nodes,
            num_remove=num_remove,
            o_v=o_v,
            o_n=o_n,
        )
        assert sc_edge_info.dtype == torch.double
        sc_edge_index = (
            torch.Tensor(sc_edge_info[:, :2]).long().t().to(edge_index.device)
        )
        # print(f"sc_edge_index shape: {sc_edge_index.shape}")
        self._is_symmetric(edge_index=sc_edge_index, num_nodes=num_nodes)

    def test_fn(self):
        for _ in range(10):
            self._test_fn()


if __name__ == "__main__":
    unittest.main()
