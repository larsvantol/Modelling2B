import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt

from graph_utils.open_graph import open_adjacency_matrix


def analyse_adjecency_matrix(
    adjacency_matrix: npt.NDArray[np.bool_], save_filename: str | None = None
) -> None:
    """
    Analyse the adjacency matrix.
    """
    # Create the graph
    graph = nx.from_numpy_array(adjacency_matrix)

    # Iterate over the nodes and get the amount of connections
    connections_per_node = []
    for node in graph.nodes:
        connections_per_node.append(graph.degree[node])

    # Sort the connections per node
    connections_per_node.sort(reverse=True)

    # Plotting the connections per node
    plt.plot(range(1, len(connections_per_node) + 1), connections_per_node)
    plt.xlabel("Node")
    plt.ylabel("Connections")
    plt.title("Amount of Connections per Node")
    plt.show()


if __name__ == "__main__":
    # Ask for the adjacency matrix
    graph_filename = askopenfilename(
        filetypes=[("CSV", "*.csv")],
        title="Open graph",
    )
    adjacency_matrix = open_adjacency_matrix(graph_filename)
    analyse_adjecency_matrix(adjacency_matrix)
