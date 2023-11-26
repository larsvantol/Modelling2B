import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from tkinter.filedialog import askopenfilename

import numpy as np
from model_classes import DataCollector, News, generate_graph_from_adjecency_and_scores
from tqdm import tqdm

from graph_utils.open_graph import open_adjacency_matrix
from graph_utils.open_scores import open_node_information_array

# Setup the simulation time
days = 6
delta_t = 1

# Define the number of nodes that will be the source of the fake news
num_sources = 5

# Ask for the adjacency matrix
# graph_filename = askopenfilename(
#     filetypes=[("CSV", "*.csv")],
#     title="Open graph",
# )
graph_filename = "C:/Users/larsv/OneDrive/Documenten/TU Delft/Vakken/Modelling 2b (AM2050-B)/Modelling-2B/tmp/Graphs/random_geometric_adjacency_matrix.csv"
adjacency_matrix = open_adjacency_matrix(graph_filename)

# Ask for the scores
# scores_filename = askopenfilename(
#     filetypes=[("CSV", "*.csv")],
#     title="Open scores",
# )
scores_filename = "C:/Users/larsv/OneDrive/Documenten/TU Delft/Vakken/Modelling 2b (AM2050-B)/Modelling-2B/tmp/Graphs/150_wappie_scores.csv"
scores = open_node_information_array(scores_filename)

# Generate the graph
graph = generate_graph_from_adjecency_and_scores(adjacency_matrix, scores)

# Set the data collector
location = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp/V2/data")
data_collector = DataCollector(os.path.dirname(os.path.abspath(__file__)))

with data_collector as data_collector:
    graph.set_data_collector(data_collector)

    # Generate the fake news
    fake_news = News(datacollector=data_collector)

    # Choose the sources
    sources = np.random.choice(list(graph.nodes), size=num_sources, replace=False)
    graph.sources = sources

    # Change the scores of the sources and add the fake news
    for source in sources:
        graph.nodes[source].wappie_score = 1
        graph.nodes[source].received_news.append(fake_news)

    graph.plot()
    data_collector.time = 1

    # Iterate over the days
    for day in tqdm(range(1, days + 1)):
        graph.plot()
        data_collector.time = day
        # Iterate over the accounts
        for id, account in graph.nodes.items():
            # Update the account
            account.use(graph.nodes)
    graph.plot()
    fake_news.add_empty_row()
