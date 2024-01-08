"""
We start with an easy model. We take a random graph, give each node a 'wappie score' and at each time step each note sends the score*connections to fake news further into the graph. The score does not change over time, and the initial scores for all nodes are taken from a normal distribution. There will be a number of nodes that will be the source of the fake news, each time step they have a certain chance of sending a fake news message into the network.
"""

import os
import pathlib
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import the graph
from tkinter.filedialog import askopenfilename
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from graph_utils.open_graph import open_adjacency_matrix


class DataCollector:
    def __init__(self, location: str):
        self.location = location
        self.time = 0
        self.data: dict[str, tuple[Any]] = {}
        self.header: dict[str, tuple[Any]] = {}

    def __enter__(self):
        return self

    def set_collection(self, collection: str) -> None:
        """Set the collection."""

        self.data[collection] = []

    def add_collection(self, collection: str, header: tuple[str]) -> None:
        """Add a collection."""

        self.data[collection] = []
        self.header[collection] = ("time", *header)

    def collect(self, data: tuple[str, tuple[Any]]) -> None:
        """Can be called at each time step to collect data."""

        self.data[data[0]].append((self.time, *data[1]))

    def __exit__(self, exc_type, exc_value, traceback):
        # Save the data
        for collection, data in self.data.items():
            # Create the header
            header = ",".join(self.header[collection])
            # Create the data
            data = "\n".join([",".join([str(item) for item in row]) for row in data])
            # Check if the folder exists otherwise create one
            pathlib.Path(self.location).mkdir(parents=True, exist_ok=True)
            # Save the data
            with open(os.path.join(self.location, f"{collection}.csv"), "w") as f:
                f.write(f"{header}\n{data}")


def save_plot_of_graph(
    graph: nx.Graph,
    output_file: str,
) -> None:
    """
    Save the graph to a file.
    """

    pos = nx.spring_layout(graph, k=1, iterations=500, seed=66)
    cmap = plt.cm.viridis
    node_colors = []
    for node in graph.nodes():
        if graph.nodes[node]["fake_news"]:
            node_colors.append("red")
            continue
        elif node in troll_nodes:
            node_colors.append("orange")
            continue
        else:
            wappie_score = graph.nodes[node]["wappie_score"]
            node_colors.append(cmap(max(wappie_score, 0)))
            continue

    pathcollection = nx.draw_networkx_nodes(
        graph,
        pos,
        # node_color=[graph.degree(node) for node in graph.nodes()], unless the node has fake news
        node_color=node_colors,
        node_size=[graph.degree(node) * 5 for node in graph.nodes()],
    )
    nx.draw_networkx_edges(graph, pos)
    plt.tight_layout()
    plt.colorbar(pathcollection)
    plt.title("Networkx Graph")
    plt.savefig(output_file)
    plt.close()


# Define the adjacency matrix
adjacency_matrix_filename = askopenfilename(
    filetypes=[("CSV", "*.csv")],
    title="Open adjacency matrix",
)
adjacency_matrix = open_adjacency_matrix(adjacency_matrix_filename)

# Define the nodes
wappie_scores_filename = askopenfilename(
    filetypes=[("CSV", "*.csv")],
    title="Open Wappie scores",
)
wappie_scores = np.loadtxt(wappie_scores_filename, dtype=float)

# Define the networkx graph
graph = nx.from_numpy_array(adjacency_matrix)

# Add the Wappie scores to the graph
for node, wappie_score in zip(graph.nodes(), wappie_scores):
    graph.nodes[node]["wappie_score"] = wappie_score
    graph.nodes[node]["fake_news"] = False

location = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp", "V1", "data"
)
data_collector = DataCollector(location=location)

plotting = False

with data_collector as datacollector:
    datacollector.add_collection("news_reached", ("news_id", "account_id"))
    datacollector.collect(("news_reached", (0, -1)))

    # Define the number of time steps
    num_time_steps = 20  # days
    delta_t = 1  # day

    # Define the number of nodes that will be the source of the fake news
    num_sources = 5

    troll_nodes = np.random.choice(
        graph.nodes(),
        size=num_sources,
        replace=False,
    )

    # Filename to save the graph
    output_filename = os.path.join(
        os.path.dirname(adjacency_matrix_filename),
        "simulation",
        "simulation",
    )

    # save_plot_of_graph(
    #     graph,
    #     f"{output_filename}_0.png",
    # )

    # Start the simulation with only the troll nodes having fake news
    # For each troll node, decide whether it will send fake news
    for troll_node in troll_nodes:
        # Send fake news to all neighbors
        for neighbor in graph.neighbors(troll_node):
            # Add the fake news messages to the neighbor
            graph.nodes[neighbor]["fake_news"] = True

    # save_plot_of_graph(
    #     graph,
    #     f"{output_filename}_1.png",
    # )

    # Run the model
    simulation_time = 0
    for time_step in tqdm(range(num_time_steps - 1)):
        data_collector.time = time_step
        # For each node, decide whether it will send fake news
        # Create a copy of the graph
        graph_copy = graph.copy()

        for node in graph.nodes():
            # Check whether the node has received fake news
            if graph.nodes[node]["fake_news"]:
                datacollector.collect(("news_reached", (0, node)))
                # The node will send fake news to the amount of neighbors times the wappie score rounded down to the nearest integer
                num_fake_news_messages = max(
                    int(graph.nodes[node]["wappie_score"] * graph.degree(node)), 0
                )  # max to prevent negative numbers
                # Send fake to random neighbors
                neighbors = list(graph.neighbors(node))
                fake_news_neighbors = np.random.choice(
                    neighbors,
                    size=int(num_fake_news_messages),
                    replace=False,
                )
                for fake_news_neighbor in fake_news_neighbors:
                    graph_copy.nodes[fake_news_neighbor]["fake_news"] = True
                # Remove the fake news from the node
                graph_copy.nodes[node]["fake_news"] = False

        # Update the graph
        graph = graph_copy.copy()

        # save_plot_of_graph(
        #     graph,
        #     f"{output_filename}_{time_step+2}.png",
        # )

        # Update the time
        simulation_time += delta_t
