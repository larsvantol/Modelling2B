"""
Creates a Wappie score for nodes in a graph
"""
import networkx as nx
import numpy as np
import numpy.typing as npt


def create_random_wappie_scores(
    num_nodes: int,
    mu: float = 0.5,
    sigma: float = 0.1,
) -> npt.NDArray[np.float_]:
    """
    Create a random Wappie score for each node in the graph.
    """

    # Create a random Wappie score for each node
    wappie_scores = np.random.normal(mu, sigma, num_nodes)

    # Make sure that the Wappie scores are between -1 and 1
    wappie_scores = np.clip(wappie_scores, -1, 1)

    return wappie_scores


def save_wappie_scores(
    wappie_scores: npt.NDArray[np.float_],
    output_file: str,
) -> None:
    """
    Save the Wappie scores to a file.
    """

    np.savetxt(output_file, wappie_scores, fmt="%f")


def read_num_of_nodes(graph: nx.Graph) -> int:
    """
    Read the number of nodes in the graph
    """
    num_nodes = graph.number_of_nodes()
    return num_nodes


if __name__ == "__main__":
    from tkinter.filedialog import askopenfilename, asksaveasfilename

    from graph_utils.open_graph import open_adjacency_matrix

    graph_filename = askopenfilename(
        filetypes=[("CSV", "*.csv")],
        title="Open graph",
    )
    adjacency_matrix = open_adjacency_matrix(graph_filename)
    graph = nx.from_numpy_array(adjacency_matrix)
    num_nodes = read_num_of_nodes(graph)
    wappie_scores = create_random_wappie_scores(num_nodes, 0, 0.5)
    file = asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv")],
        title="Save Wappie scores",
        initialfile=f"{num_nodes}_wappie_scores.csv",
    )
    save_wappie_scores(wappie_scores, file)
