import matplotlib.pyplot as plt
import numpy as np


def plot_scores(scores, title, xlabel, ylabel):
    plt.hist(scores, bins=100)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == "__main__":
    from tkinter.filedialog import askopenfilename

    scores_filename = askopenfilename(
        filetypes=[("CSV", "*.csv")],
        title="Open scores",
    )

    scores = np.loadtxt(scores_filename)

    plot_scores(scores, "Wappie scores", "Wappie score", "Number of nodes")
