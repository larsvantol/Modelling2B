"""
Classes for the account and news objects.
"""

from __future__ import annotations

from typing import Self, Any

import numpy as np
import numpy.typing as npt

from tqdm import tqdm

class News:
    def __init__(self) -> None:
        self.id: int = self.get_next_id()
        self.reached_accounts: list[int] = []

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

    @classmethod
    def get_next_id(cls):
        """Return the next available ID."""

        if not hasattr(cls, "_id_counter"):
            cls._id_counter = 0
        else:
            cls._id_counter += 1
        return cls._id_counter

    def send_news(self, news: News, account: Self) -> None:
        """Send news to another account."""

        self.sent_news.append(news)
        account.receive_news(news)

    def receive_news(self, news: News) -> None:
        """Receive news from another account."""

        self.received_news.append(news)

    def decide_to_send_news(self, news: News) -> None:
        """Decide whether to use the news or not, based on the wappie score."""

        num_fake_news_messages = max(
            int(self.wappie_score * self.degree()), 0
        )  # max to prevent negative numbers
        # Send fake news to random neighbors
        fake_news_neighbors = np.random.choice(
            list(self.connections),
            size=int(num_fake_news_messages),
            replace=False,
        )

        for neighbor in fake_news_neighbors:
            self.send_news(news, neighbor)

    def use(self) -> None:
        """Use the account."""

        # Iterate over the news that the account has received this time step
        for received_news in self.received_news:
            # Check whether the account has not received the news before
            # If it has, it will not send it again
            if received_news not in self.past_news:
                # Add the news to the news that the account has seen
                self.past_news.append(received_news)
                # Decide whether to send the news to its neighbors
                self.decide_to_send_news(received_news)


class Graph:
    def __init__(self, nodes: dict[int, Account]) -> None:
        self.nodes = nodes
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

    def collect(self, data: Any) -> None:
        """Can be called at each time step to collect data."""

        print(data)