"""
Simple python script for explaing how to use networkx
"""
import os

import matplotlib.pyplot as plt
import networkx as nx

##########################################################
# Create a simple graph
##########################################################

# Create a graph
Graph = nx.Graph()

# Add nodes
Graph.add_nodes_from(range(1, 9))
Graph.add_node(10)

# Add edges with weights
Graph.add_edges_from(
    [
        (1, 2, {"weight": 0.125}),
        (1, 3, {"weight": 0.75}),
        (1, 4, {"weight": 0.375}),
        (2, 5, {"weight": 1.0}),
        (2, 6, {"weight": 0.5}),
        (3, 7, {"weight": 0.125}),
        (3, 8, {"weight": 0.75}),
        (4, 9, {"weight": 0.375}),
        (4, 10, {"weight": 1.0}),
        (2, 10, {"weight": 0.5}),
        (5, 7, {"weight": 0.125}),
        (6, 9, {"weight": 0.375}),
    ]
)

##########################################################
# Plot the graph
##########################################################

# Draw the graph,
# use different color for each weight
# and use different color for each node depending on its degree
nx.draw(
    Graph,
    with_labels=True,
    node_color=[Graph.degree(node) for node in Graph.nodes()],
    edge_color=[Graph.get_edge_data(edge[0], edge[1])["weight"] for edge in Graph.edges()],
    cmap=plt.cm.viridis,
)
plt.show()


##########################################################
# Save the graph
##########################################################

# Write the graph to a file

# Get directory of this file
dir_path = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(dir_path, "tmp", "example", "simple_graph.gexf")

# Write the graph to a file
nx.write_gexf(Graph, file)

##########################################################
# Read the graph
##########################################################

# Read the graph from a file
Graph2 = nx.read_gexf(file)

# Print the nodes
print("Nodes: ", Graph2.nodes())

##########################################################
# Algorithms
##########################################################

# There are many algorithms available in networkx, see:
# https://networkx.github.io/documentation/stable/reference/algorithms/index.html

# Get the shortest path between two nodes
shortest_path = nx.shortest_path(Graph, 1, 10, weight="weight")
print("Shortest path between 1 and 10: ", shortest_path)
