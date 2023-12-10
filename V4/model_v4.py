import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tkinter.filedialog import askopenfilename

import numpy as np
from model_classes import DataCollector, News, generate_graph_from_adjecency_and_scores
from tqdm import tqdm

from graph_utils.open_graph import open_adjacency_matrix
from graph_utils.open_scores import open_node_information_array

# Setup the simulation time
days = 200
delta_t = 1

# Define the number of nodes that will be the source of the fake news
num_sources = 50

# Ask for the adjacency matrix
graph_filename = askopenfilename(
    filetypes=[("CSV", "*.csv")],
    title="Open graph",
)
# graph_filename = "C:/Users/larsv/OneDrive/Documenten/TU Delft/Vakken/Modelling 2b (AM2050-B)/Modelling-2B/tmp/Graphs/150_random_geometric_adjacency_matrix.csv"
print("Opening graph")
adjacency_matrix = open_adjacency_matrix(graph_filename)
print("Graph opened")

# Ask for the scores
scores_filename = askopenfilename(
    filetypes=[("CSV", "*.csv")],
    title="Open scores",
)
# scores_filename = "C:/Users/larsv/OneDrive/Documenten/TU Delft/Vakken/Modelling 2b (AM2050-B)/Modelling-2B/tmp/Graphs/150_wappie_scores.csv"
print("Opening scores")
scores = open_node_information_array(scores_filename)
print("Scores opened")

# Generate the graph
print("Generating graph")
graph = generate_graph_from_adjecency_and_scores(adjacency_matrix, scores)

# Set the data collector
location = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp", "V4", "data"
)
data_collector = DataCollector(location=location)

plotting = False

with data_collector as data_collector:
    print("Setting data collector")
    graph.set_data_collector(data_collector)

    # Generate the fake news
    print("Generating fake news")
    fake_news = News(datacollector=data_collector)

    # Choose the sources
    print("Choosing sources")
    # sources = np.random.choice(list(graph.nodes), size=num_sources, replace=False)
    sources = [
        304,
        633,
        909,
        1313,
        950,
        596,
        691,
        871,
        827,
        283,
        132,
        336,
        1062,
        967,
        949,
        603,
        1478,
        1483,
        812,
        1236,
        1026,
        91,
        1446,
        1499,
        1001,
        1401,
        195,
        546,
        386,
        891,
        1150,
        53,
        128,
        593,
        614,
        407,
        300,
        881,
        1268,
        634,
        1339,
        672,
        560,
        876,
        720,
        217,
        219,
        6,
        1393,
        1442,
    ]
    print(sources)
    graph.sources = sources

    # Change the scores of the sources and add the fake news
    for source in sources:
        graph.nodes[source].change_to_source()
        graph.nodes[source].received_news.append((graph.nodes[source], fake_news))

    # Plot the graph
    print("Plotting graph")
    print(f"Day {data_collector.time}")
    if plotting:
        graph.plot()

    # Iterate over the days
    print("Iterating over the days")
    for day in tqdm(range(1, days + 1)):
        data_collector.time = day
        # Iterate over the accounts
        # for id, account in tqdm(graph.nodes.items()):
        for id, account in graph.nodes.items():
            # Update the account
            start = time.time_ns()
            recieved_news_list = account.received_news
            account.use(graph.nodes)
            end = time.time_ns()
            # if end - start > 1e6 and not account.source:
            #     print(f"Account {id} took {(end - start) / 1e6} ms")
            #     print(f"\tWappie score:\t{account.wappie_score}")
            #     print(f"\tConnections:\t{account.connections}")
            #     print(f"\tNews received:\t{recieved_news_list}")
            #     print(f"\tNews sent:\t{account.sent_news}")
            #     print(f"\tPast news:\t{account.past_news}")
        # print(f"Day {data_collector.time}")
        if plotting:
            graph.plot()
        fake_news.add_empty_row()
    fake_news.add_empty_row()
    if "news_reached" in data_collector.data:
        for datapoint in data_collector.data["news_reached"]:
            print(datapoint)
