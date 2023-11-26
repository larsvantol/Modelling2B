import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np

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


def main():
    # Ask for the data file
    news_reached_filename = askopenfilename(
        filetypes=[("CSV", "*.csv")],
        title="Open news reached file",
    )

    # Open the data file
    news_reached = np.loadtxt(news_reached_filename, delimiter=",", skiprows=1)
    headers = np.loadtxt(news_reached_filename, delimiter=",", max_rows=1, dtype=str)

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

    # Set the labels
    plt.xlabel("Time")
    plt.ylabel("Accounts reached")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
