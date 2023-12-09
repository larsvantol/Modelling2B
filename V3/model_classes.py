"""
Classes for the account and news objects.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt


class News:
    def __init__(self, datacollector: DataCollector) -> None:
        self.id: int = self.get_next_id()
        self.reached_accounts: list[int] = []
        self.datacollector = datacollector
        datacollector.add_collection("news_reached", ("news_id", "account_id"))
        self.add_empty_row()

    def add_empty_row(self) -> None:
        """Add an empty row to the data collector."""

        self.datacollector.collect(("news_reached", (self.id, -1)))

    def mark_as_reached(self, account: Account) -> None:
        """Mark the news as reached by an account."""

        self.reached_accounts.append(account.id)
        self.datacollector.collect(("news_reached", (self.id, account.id)))

    @classmethod
    def get_next_id(cls) -> int:
        """Return the next available ID."""

        if not hasattr(cls, "_id_counter"):
            cls._id_counter = 0
        else:
            cls._id_counter += 1
        return cls._id_counter


NewsList = list[News]


class Account:
    def __init__(self, wappie_score: float, connections: list[int]) -> None:
        self.id = self.get_next_id()

        self.wappie_score = wappie_score

        self.connections = [connection for connection in connections if connection != self.id]

        self.received_news: NewsList = []

        self.past_news: NewsList = []
        self.sent_news: NewsList = []

    def set_data_collector(self, data_collector: DataCollector) -> None:
        """Set the data collector."""

        self.data_collector = data_collector

    def degree(self) -> int:
        """Return the degree of the account."""

        return len(self.connections)

    def believes(self, news: News, source: Account) -> bool:
        """
        Returns if the account believes the fake news
        """
        beta = -1
        while beta < 0:
            beta = abs(np.random.normal(0.1, 0.075))

        # The account believes news comming from a source with a similar wappie score  own score +- beta
        return abs(self.wappie_score - source.wappie_score) <= beta

    @classmethod
    def get_next_id(cls):
        """Return the next available ID."""

        if not hasattr(cls, "_id_counter"):
            cls._id_counter = 0
        else:
            cls._id_counter += 1
        return cls._id_counter

    def send_news(self, news: News, account: Account) -> None:
        """Send news to another account."""

        self.sent_news.append(news)
        account.receive_news(news)

    def receive_news(self, news: News) -> None:
        """Receive news from another account."""

        self.received_news.append(news)

    def whom_to_send_news(self, news: News, nodes: dict[int, Account]) -> npt.NDArray[np.int_]:
        """Decide whether to use the news or not, based on the wappie score."""

        # Check whether the account believes the news
        if not self.believes(news, nodes[news.reached_accounts[-1]]):
            return np.array([], dtype=int)
        # So the account believes the news and will send it to its neighbors

        fake_news_neighbors = np.array([connection for connection in self.connections], dtype=int)

        return fake_news_neighbors

    def use(self, nodes: dict[int, Account]) -> None:
        """Use the account."""

        # Iterate over the news that the account has received this time step
        for received_news in self.received_news:
            # Check whether the account has not received the news before
            # If it has, it will not send it again
            if received_news not in self.past_news:
                # Mark the news as reached
                received_news.mark_as_reached(self)
                # Add the news to the news that the account has seen
                self.past_news.append(received_news)
                # Decide whether to send the news to its neighbors
                send_list = self.whom_to_send_news(news=received_news, nodes=nodes)
                # Send the news to the neighbors
                for id in send_list:
                    self.send_news(received_news, nodes[id])
        # Empty the received news list
        self.received_news = []


class Graph:
    def __init__(self, nodes: dict[int, Account]) -> None:
        self.nodes = nodes
        self.sources = np.array([], dtype=int)
        self.num_nodes = len(nodes)

    def adjecency_matrix(self) -> npt.NDArray[np.int_]:
        """Return the adjecency matrix of the graph."""

        adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        for id, node in self.nodes.items():
            for connection_id in node.connections:
                adjacency_matrix[id, connection_id] = 1
        return adjacency_matrix

    def set_data_collector(self, data_collector: DataCollector) -> None:
        """Set the data collector."""

        self.data_collector = data_collector

        for node in self.nodes.values():
            node.set_data_collector(data_collector)

    def plot(self) -> None:
        """Plot the graph."""

        # Create the graph
        G = nx.Graph()
        for id, node in self.nodes.items():
            G.add_node(id, wappie_score=node.wappie_score, received_news=node.received_news)
        for id, node in self.nodes.items():
            for connection_id in node.connections:
                G.add_edge(id, connection_id)

        # Save the graph
        self.save_plot_of_graph(
            G,
            os.path.join(self.data_collector.location, f"graph_{self.data_collector.time}.png"),
        )

    def save_plot_of_graph(
        self,
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
            if graph.nodes[node]["received_news"] != []:
                node_colors.append("red")
                continue
            elif node in self.sources:
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
        # Check if the folder exists otherwise create one
        pathlib.Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
        plt.close()

    def __str__(self) -> str:
        return f"Graph with {self.num_nodes} nodes."


def generate_graph_from_adjecency_and_scores(
    adjecency_matrix: npt.NDArray[np.bool_], wappie_scores: npt.NDArray[np.float_]
) -> Graph:
    """Generate a graph from an adjecency matrix and wappie scores."""

    # Create the nodes
    accounts = {}
    for i in range(len(adjecency_matrix)):
        connections = [id for id, connection in enumerate(adjecency_matrix[i]) if connection]
        accounts[i] = Account(wappie_score=wappie_scores[i], connections=connections)

    # Create the graph
    return Graph(accounts)


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
