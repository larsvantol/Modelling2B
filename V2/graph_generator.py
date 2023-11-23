import sys

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.getcwd())
# pylint: enable=wrong-import-position

from tkinter.filedialog import askopenfilename

import numpy as np
from model_classes import Account, Graph, News

from open_graph import open_adjacency_matrix

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

# Create a list of accounts
network = Graph.generate_from_adjecency_and_scores(adjacency_matrix, wappie_scores)
print(network.adjecency_matrix())
for account in network.nodes.values():
    print(account.wappie_score)
