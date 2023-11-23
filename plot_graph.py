from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import networkx as nx

from graph_utils.open_graph import open_adjacency_matrix

# Define the adjacency matrix
matrix = askopenfilename(
    filetypes=[("CSV", "*.csv")],
    title="Open adjacency matrix",
)
adjacency_matrix = open_adjacency_matrix(matrix)

# Plot the adjacency matrix
# plt.imshow(adjacency_matrix, cmap="binary")
# plt.colorbar()
# plt.title("Adjacency Matrix")
# plt.show()

# Define the networkx graph
graph = nx.from_numpy_array(adjacency_matrix)

# Plot the networkx graph

pos = nx.spring_layout(graph, k=1, iterations=500, seed=66)
pathcollection = nx.draw_networkx_nodes(
    graph,
    pos,
    node_color=[graph.degree(node) for node in graph.nodes()],
    node_size=[graph.degree(node) * 5 for node in graph.nodes()],
    cmap=plt.cm.viridis,
)
nx.draw_networkx_edges(graph, pos)
plt.tight_layout()
plt.colorbar(pathcollection)
plt.title("Networkx Graph")
plt.show()
