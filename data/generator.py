"""A helper script to generate adjacency matrix data"""
import imp
import networkx as nx

def generate_complete_graph(N, out_file):
    """generate adjacency matrix for a complete graph
    with N nodes and unit weights. The tsv values are stored
    in the `out_file`.

    The content of the `out_file` can then be read as
    (row, col, value) tuples to build the matrix. This format
    simplifies the creation of sparse matrices.

    Args:
        N: Number of nodes in the graph
        out_file: file path to store the data

    Returns:
        None
    """
    res = []
    for i in range(N):
        for j in range(i, N):
            if i == j:
                val = 0
                res.append("{}\t{}\t{}\n".format(i+1, j+1, val))
            else:
                val = 1
                res.append("{}\t{}\t{}\n".format(i+1, j+1, val))
                res.append("{}\t{}\t{}\n".format(j+1, i+1, val))
    with open(out_file, "w+") as f:
        f.writelines(res)

def generate_grid_graph(dims, out_file):
    """generate adjacency matrix for a grid_graph(dims)
    and unit weights. The tsv values are stored
    in the `out_file`.

    The content of the `out_file` can then be read as
    (row, col, value) tuples to build the matrix. This format
    simplifies the creation of sparse matrices.

    Args:
        dims: A list of dimensions for the grid
        out_file: file path to store the data

    Returns:
        None
    """
    res = []
    G = nx.grid_graph(dims)
    A = nx.adjacency_matrix(G)
    rows, cols = A.nonzero()
    for r, c in zip(rows, cols):
        res.append("{}\t{}\t{}\n".format(r+1, c+1, 1))
    with open(out_file, "w+") as f:
        f.writelines(res)


if __name__ == "__main__":
    for i in [5, 10, 20, 50, 100, 500, 1000, 2000, 5000]:
        out_file = "connected{}.tsv".format(i)
        generate_complete_graph(i, out_file)

    for i in [[3,3,3], [10,10,10], [20,20,20], [50,50,50]]:
        out_file = "grid{}.tsv".format(i[0])
        generate_grid_graph(i, out_file)
