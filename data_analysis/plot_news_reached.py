import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tkinter.filedialog import askopenfilename, asksaveasfilename

import matplotlib.pyplot as plt
import numpy as np
from get_simulation_data import get_simulation_data

# Example of csv:

# time,news_id,account_id
# 0,0,3
# 0,0,15
# 0,0,40
# 0,0,42
# 0,0,45
# 0,0,53
# 0,0,60
# 0,0,82


def plot_news_reached(
    csv_filename: str | None = None,
    graph_filename: str | None = None,
    simulation_data: str | None = None,
):
    # Ask for the data file
    if not csv_filename:
        csv_filename = askopenfilename(
            filetypes=[("CSV", "*.csv")],
            title="Open news reached file",
        )

    # Remove the extension of the graph_filename
    if graph_filename:
        graph_filename = os.path.splitext(graph_filename)[0]

    # Open the data file
    news_reached = np.loadtxt(csv_filename, delimiter=",", skiprows=1)
    headers = np.loadtxt(csv_filename, delimiter=",", max_rows=1, dtype=str)

    # Get the news ids
    news_ids = np.unique(news_reached[:, 1])

    # Get the time
    time = np.unique(news_reached[:, 0])

    # Get the number of accounts reached per news id per timestep
    news_reached_per_id = {}
    for news_id in news_ids:
        news_reached_per_id[news_id] = []
        for timestep in range(len(time)):
            news_reached_per_id[news_id].append(
                len(
                    list(
                        filter(
                            lambda x: x[0] == timestep and x[1] == news_id and x[2] != -1,
                            news_reached,
                        )
                    )
                )
            )

    # Plot the data
    plt.figure()
    for news_id in news_ids:
        print(news_reached_per_id[news_id])
        plt.plot(time, news_reached_per_id[news_id], label=f"News {news_id}")

    # Set the axis
    plt.xlabel("Time")
    plt.xlim(0, time[-1])
    plt.ylabel("Accounts reached")
    plt.ylim(0, simulation_data["population"])
    plt.legend()
    plt.grid()
    if graph_filename:
        plt.savefig(
            f"{graph_filename}_absolute.png",
            dpi=300,
            format="png",
            bbox_inches="tight",
            transparent=False,
        )
    else:
        plt.show()

    # Now for the percentage of accounts reached
    total_accounts = int(simulation_data["population"])
    plt.figure()
    for news_id in news_ids:
        plt.plot(
            time,
            np.array(news_reached_per_id[news_id]) / total_accounts,
            label=f"News {news_id}",
        )

    # Set the axis
    plt.xlabel("Time")
    plt.xlim(0, time[-1])
    plt.ylabel("Accounts reached (%)")
    plt.ylim(0, 1)
    plt.grid()
    plt.legend()
    if graph_filename:
        plt.savefig(
            f"{graph_filename}_percentage.png",
            dpi=300,
            format="png",
            bbox_inches="tight",
            transparent=False,
        )
    else:
        plt.show()

    # Now for the cummulative number of accounts reached
    plt.figure()
    for news_id in news_ids:
        plt.plot(
            time,
            np.cumsum(news_reached_per_id[news_id]) / total_accounts,
            label=f"News {news_id}",
        )

    # Set the axis
    plt.xlabel("Time")
    plt.xlim(0, time[-1])
    plt.ylabel("Cumulative number of accounts reached (%)")
    plt.ylim(0, 1)
    plt.grid()
    plt.legend()

    if graph_filename:
        plt.savefig(
            f"{graph_filename}_cumulative.png",
            dpi=300,
            format="png",
            bbox_inches="tight",
            transparent=False,
        )
    else:
        plt.show()


if __name__ == "__main__":
    news_reached_filename = askopenfilename(
        filetypes=[("CSV", "*.csv")],
        title="Open news reached file",
    )
    graph_filename = asksaveasfilename(
        filetypes=[("PNG", "*.png")],
        title="Save graph",
        defaultextension=".png",
        initialfile="news_reached_graph.png",
    )
    simulation_data = get_simulation_data(os.path.dirname(news_reached_filename))
    plot_news_reached(
        csv_filename=news_reached_filename,
        graph_filename=graph_filename,
        simulation_data=simulation_data,
    )
