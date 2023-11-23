import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_classes import DataCollector, Account, News, Graph, generate_graph_from_adjecency_and_scores
from graph_utils.open_graph import open_adjacency_matrix
from graph_utils.open_scores import open_node_information_array
from tkinter.filedialog import askopenfilename

from tqdm import tqdm
import numpy as np

# Setup the simulation time
days = 100
delta_t = 1

# Define the number of nodes that will be the source of the fake news
num_sources = 2

# Ask for the adjacency matrix
graph_filename = askopenfilename(
        filetypes=[("CSV", "*.csv")],
        title="Open graph",
    )
adjacency_matrix = open_adjacency_matrix(graph_filename)

# Ask for the scores
scores_filename = askopenfilename(
        filetypes=[("CSV", "*.csv")],
        title="Open scores",
    )
scores = open_node_information_array(scores_filename)

# Generate the graph
graph = generate_graph_from_adjecency_and_scores(adjacency_matrix, scores)

# Set the data collector
location = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp/V2/data")
data_collector = DataCollector(os.path.dirname(os.path.abspath(__file__)))
graph.set_data_collector(data_collector)

# Generate the fake news
fake_news = News()

# Generate the sources
sources: list[Account] = []
# Choose the sources
for i in range(num_sources):
    # Choose a random account
    source = graph.nodes[np.random.randint(0, len(graph.nodes))]
    # Change the wappie score of the source to 1
    source.wappie_score = 1
    # Add the source to the list of sources
    sources.append(source)

# Iterate over the days

for day in tqdm(range(days)):