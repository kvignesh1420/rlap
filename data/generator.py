"""A helper script to generate adjacency matrix data"""
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

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


def main():
    connected_nodes = [5, 10, 20, 50, 100, 500, 1000, 2000, 5000]
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(generate_complete_graph, c, "connected{}.tsv".format(c)): i for i, c in enumerate(connected_nodes)}
        for future in as_completed(future_map):
            i = future_map[future]
            future.result()
            print("generated connected graph with: {} nodes".format(connected_nodes[i]))
    
    grid_nodes = [[3,3,3], [10,10,10], [20,20,20], [50,50,50], [100, 100, 100]]
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(generate_grid_graph, c, "grid{}.tsv".format(c[0])): i for i, c in enumerate(grid_nodes)}
        for future in as_completed(future_map):
            i = future_map[future]
            future.result()
            print("generated grid graph with: {} nodes".format(grid_nodes[i]))


if __name__ == "__main__":
    main()
