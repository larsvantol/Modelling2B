"""
Classes for the account and news objects.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import numpy.typing as npt


class News:
    def __init__(self) -> None:
        self.id: int = self.get_next_id()

    @classmethod
    def get_next_id(cls) -> int:
        """Return the next available ID."""

        if not hasattr(cls, "_id_counter"):
            cls._id_counter = 0
        else:
            cls._id_counter += 1
        return cls._id_counter


class Account:
    def __init__(self, wappie_score: float, connections: list[int]) -> None:
        self.id = self.get_next_id()

        self.wappie_score = wappie_score

        self.connections = [connection for connection in connections if connection != self.id]

        self.received_news: list[News] = []
        self.past_news: list[News] = []
        self.sent_news: list[News] = []

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
        # Send fake to random neighbors
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

    @classmethod
    def generate_from_adjecency_and_scores(
        cls, adjecency_matrix: npt.NDArray[np.bool_], wappie_scores: npt.NDArray[np.float_]
    ) -> Self:
        """Generate the graph from an adjecency matrix and wappie scores."""

        # Create the nodes
        accounts = {}
        for i in range(len(adjecency_matrix)):
            connections = [id for id, connection in enumerate(adjecency_matrix[i]) if connection]
            accounts[i] = Account(wappie_score=wappie_scores[i], connections=connections)

        # Create the graph
        return Graph(accounts)
