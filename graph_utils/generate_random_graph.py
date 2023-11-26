from tkinter.filedialog import asksaveasfilename
from tkinter.simpledialog import askinteger

import networkx as nx
import numpy as np
import numpy.typing as npt


def generate_random_geometric_adjacency_matrix(
    num_nodes: int,
    connection_probability: float = 0.125,
) -> npt.NDArray[np.bool_]:
    graph = nx.random_geometric_graph(150, 0.125)
    adjacency_matrix = nx.to_numpy_array(graph)
    return adjacency_matrix


def generate_random_adjacency_matrix(
    num_nodes: int,
    connection_probability: float = 0.5,
) -> npt.NDArray[np.bool_]:
    """
    Generate a random adjacency matrix for a graph with `num_nodes` nodes.
    """

    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                random_number = np.random.rand()
                if random_number < connection_probability:
                    adjacency_matrix[i][j] = 1

    return adjacency_matrix


def save_adjacency_matrix(
    adjacency_matrix: npt.NDArray[np.bool_],
    output_file: str,
) -> None:
    """
    Save the adjacency matrix to a file.
    """

    np.savetxt(output_file, adjacency_matrix, fmt="%d")


if __name__ == "__main__":
    num_nodes = None
    while num_nodes is None:
        num_nodes = askinteger(
            "Number of nodes",
            "How many nodes should the graph have?",
            minvalue=1,
            maxvalue=1000,
        )
    adjacency_matrix = generate_random_geometric_adjacency_matrix(int(num_nodes), 0.15)
    file = asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv")],
        title="Save adjacency matrix",
        initialfile="random_geometric_adjacency_matrix.csv",
    )
    save_adjacency_matrix(adjacency_matrix, file)
