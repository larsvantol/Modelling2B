"""
Classes for the account and news objects.
"""
from __future__ import annotations

import os
import pathlib
import time
from typing import Any, Self

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from tqdm import tqdm


class News:
    def __init__(self, datacollector: DataCollector) -> None:
        self.id: int = self.get_next_id()
        self.reached_accounts: list[int] = []
        self.believed_accounts: list[int] = []
        self.datacollector = datacollector
        datacollector.add_collection("news_reached", ("news_id", "account_id"))
        self.add_empty_row()

    def add_empty_row(self) -> None:
        """Add an empty row to the data collector."""

        self.datacollector.collect(("news_reached", (self.id, -1)))

    def mark_as_believed(self, account: Account) -> None:
        """Mark the news as reached by an account."""

        self.reached_accounts.append(account.id)
        self.datacollector.collect(("news_reached", (self.id, account.id)))

    def __str__(self) -> str:
        return f"News {self.id}"

    def __repr__(self) -> str:
        return f"News {self.id}"

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
        self.source = False

        self.connections = [connection for connection in connections if connection != self.id]

        self.received_news: list[(Self, News)] = []

        self.sigma = 0

        self.past_news: NewsList = []
        self.sent_news: NewsList = []

    def __str__(self) -> str:
        return f"Account {self.id}"

    def __repr__(self) -> str:
        return f"Account {self.id}"

    def set_data_collector(self, data_collector: DataCollector) -> None:
        """Set the data collector."""

        self.data_collector = data_collector

    def degree(self) -> int:
        """Return the degree of the account."""

        return len(self.connections)

    def calculate_neighbours_average_believe(self, news: News, nodes: dict[int, Account]) -> float:
        """Calculate the average wappie score of the neighbors."""
        neighbor_believes = [
            1
            if news in nodes[connection].sent_news  # an account believes the news if it has sent it
            else 0
            for connection in self.connections
        ]
        return np.mean(neighbor_believes)

    def believes(self, news: News, source: Account, nodes: dict[int, Account]) -> bool:
        """
        Returns if the account believes the fake news
        """
        sigma = self.sigma
        alpha = np.clip(abs(np.random.normal(abs(self.wappie_score), sigma)), 0, 1)

        # The account believes news comming from a source with a similar wappie score  own score +- beta
        # print(
        #     f"Account {self.id} has a \n\twappie score\t{abs(self.wappie_score)}\n\talpha\t{alpha} \n\t(1-alpha)\t{1-alpha} \n\t believe:\t{self.calculate_neighbours_average_believe(news, nodes)}"
        # )
        # print()
        return self.calculate_neighbours_average_believe(news, nodes) >= (1 - alpha)

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

        if news not in self.sent_news:
            self.sent_news.append(news)
        account.receive_news((self, news))

    def receive_news(self, news: News) -> None:
        """Receive news from another account."""

        self.received_news.append(news)

    def change_to_source(self) -> None:
        """Change the account to a source."""

        self.wappie_score = 1
        self.source = True

        def new_use(nodes):
            for source, received_news in self.received_news:
                # Check if the news has been sent before by the account, if so, skip it as it will not be sent again
                has_been_sent = False
                for news in self.sent_news:
                    if news.id == received_news.id:
                        has_been_sent = True
                        break
                if has_been_sent:
                    continue
                send_list = np.array([connection for connection in self.connections], dtype=int)
                for id in send_list:
                    self.send_news(received_news, nodes[id])
            self.sent_news = []

        self.use = new_use

    def whom_to_send_news(
        self, news: News, source: Account, nodes: dict[int, Account]
    ) -> npt.NDArray[np.int_]:
        """Decide whether to use the news or not, based on the wappie score."""

        # Check whether the account believes the news
        if not self.believes(news, source, nodes):
            return np.array([], dtype=int)
        # So the account believes the news and will send it to its neighbors
        news.mark_as_believed(self)

        fake_news_neighbors = np.array([connection for connection in self.connections], dtype=int)

        return fake_news_neighbors

    def use(self, nodes: dict[int, Account]) -> None:
        """Use the account."""

        # Iterate over the news that the account has received this time step
        for source, received_news in self.received_news:
            # Check if the news has been sent before by the account, if so, skip it as it will not be sent again
            has_been_sent = False
            for news in self.sent_news:
                if news.id == received_news.id:
                    has_been_sent = True
                    break
            if has_been_sent:
                continue

            # Decide whether to send the news to its neighbors
            start = time.time_ns()
            send_list = self.whom_to_send_news(news=received_news, source=source, nodes=nodes)
            end = time.time_ns()
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
            G.add_node(
                id,
                wappie_score=node.wappie_score,
                received_news=node.received_news,
                sent_news=node.sent_news,
            )
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
            if node in self.sources:
                node_colors.append("orange")
                continue
            if graph.nodes[node]["received_news"] != []:
                if graph.nodes[node]["sent_news"] != []:
                    node_colors.append("pink")
                    continue
                node_colors.append("red")
                continue
            if graph.nodes[node]["sent_news"] != []:
                node_colors.append("green")
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
    for i in tqdm(range(len(adjecency_matrix))):
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
