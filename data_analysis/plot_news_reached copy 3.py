import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.simpledialog import askstring

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


def parse_data(csv_filename: str) -> np.ndarray:
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
    return time, news_ids, news_reached_per_id


def plot_news_reached(
    csv_filename: str,
    graph_filename: str,
    simulation_data: str,
    model_name: str = "Model",
):
    # Remove the extension of the graph_filename
    if graph_filename:
        graph_filename = os.path.splitext(graph_filename)[0]

    # Open the data file
    time, news_ids, news_reached_per_id = parse_data(csv_filename)

    # Plot the data
    plt.figure()
    for news_id in news_ids:
        print(news_reached_per_id[news_id])
        plt.plot(time, news_reached_per_id[news_id], label=f"{model_name}")

    # Set the axis
    plt.xlabel("Time")
    plt.xlim(0, time[-1])
    plt.ylabel("Accounts reached")
    plt.ylim(0, simulation_data["population"])
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
            label=f"{model_name}",
        )

    # Set the axis
    plt.xlabel("Time")
    plt.xlim(0, time[-1])
    plt.ylabel("Accounts reached (%)")
    plt.ylim(0, 1)
    plt.grid()
    # plt.legend()
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
            label=f"{model_name}",
        )

    # Set the axis
    plt.xlabel("Time")
    plt.xlim(0, time[-1])
    plt.ylabel("Cumulative number of accounts reached (%)")
    plt.ylim(0, 1)
    plt.grid()
    # plt.legend()

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


def plot_news_reached_2(
    csv_filename: str,
    csv_filename_2: str,
    graph_filename: str,
    simulation_data: str,
    simulation_data_2: str,
    model_name: str = "Model",
    model_name_2: str = "Previous model",
):
    # Remove the extension of the graph_filename
    if graph_filename:
        graph_filename = os.path.splitext(graph_filename)[0]

    # Open the data file
    time, news_ids, news_reached_per_id = parse_data(csv_filename)
    time_2, news_ids_2, news_reached_per_id_2 = parse_data(csv_filename_2)

    # Plot the data
    plt.figure()
    for news_id in news_ids:
        print(news_reached_per_id[news_id])
        plt.plot(time, news_reached_per_id[news_id], label=f"{model_name}")
        plt.plot(time_2, news_reached_per_id_2[news_id], label=f"{model_name_2}", color="silver")

    # Set the axis
    plt.xlabel("Time")
    plt.xlim(0, max(time[-1], time_2[-1]))
    plt.ylabel("Accounts reached")
    plt.ylim(0, max(simulation_data["population"], simulation_data_2["population"]))
    # plt.legend()
    plt.grid()
    if graph_filename:
        plt.savefig(
            f"{graph_filename}_absolute_with_old.png",
            dpi=300,
            format="png",
            bbox_inches="tight",
            transparent=False,
        )
    else:
        plt.show()

    # Now for the percentage of accounts reached
    total_accounts = int(simulation_data["population"])
    total_accounts_2 = int(simulation_data_2["population"])
    plt.figure()
    for news_id in news_ids:
        plt.plot(
            time,
            np.array(news_reached_per_id[news_id]) / total_accounts,
            label=f"{model_name}",
        )
        plt.plot(
            time_2,
            np.array(news_reached_per_id_2[news_id]) / total_accounts_2,
            label=f"{model_name_2}",
            color="silver",
        )

    # Set the axis
    plt.xlabel("Time")
    plt.xlim(0, max(time[-1], time_2[-1]))
    plt.ylabel("Accounts reached (%)")
    plt.ylim(0, 1)
    plt.grid()
    # plt.legend()
    if graph_filename:
        plt.savefig(
            f"{graph_filename}_percentage_with_old.png",
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
            label=f"{model_name}",
        )
        plt.plot(
            time_2,
            np.cumsum(news_reached_per_id_2[news_id]) / total_accounts_2,
            label=f"{model_name_2}",
            color="silver",
        )

    # Set the axis
    plt.xlabel("Time")
    plt.xlim(0, max(time[-1], time_2[-1]))
    plt.ylabel("Cumulative number of accounts reached (%)")
    plt.ylim(0, 1.01)
    plt.grid()
    # plt.legend()

    if graph_filename:
        plt.savefig(
            f"{graph_filename}_cumulative_with_old.png",
            dpi=300,
            format="png",
            bbox_inches="tight",
            transparent=False,
        )
    else:
        plt.show()


def plot_news_reached_3(
    csv_filenames: list[str],
    graph_filename: str,
    simulation_data: list[str],
    model_names: list[str],
    legend: bool = True,
):
    # Remove the extension of the graph_filename
    if graph_filename:
        graph_filename = os.path.splitext(graph_filename)[0]

    # Open the data file
    max_population = 1
    max_time = 1
    data = {}
    for csv_filename, single_simulation_data, model_name in zip(
        csv_filenames, simulation_data, model_names
    ):
        time, news_ids, news_reached_per_id = parse_data(csv_filename)
        data[model_name] = {
            "time": time,
            "news_ids": news_ids,
            "news_reached_per_id": news_reached_per_id,
            "simulation_data": single_simulation_data,
        }
        max_population = max(max_population, int(single_simulation_data["population"]))
        max_time = max(max_time, time[-1])

    # # Plot the data
    # plt.figure()
    # for model in data:
    #     time = data[model]["time"]
    #     news_reached_per_id = data[model]["news_reached_per_id"]
    #     for news_id in news_ids:
    #         plt.plot(time, news_reached_per_id[news_id], label=f"{model}")

    # # Set the axis
    # plt.xlabel("Time")
    # plt.xlim(0, max_time)
    # plt.ylabel("Accounts reached")
    # plt.ylim(0, max_population)
    # if legend:
    #     plt.legend()
    # plt.grid()
    # if graph_filename:
    #     plt.savefig(
    #         f"{graph_filename}_absolute_with_old.png",
    #         dpi=300,
    #         format="png",
    #         bbox_inches="tight",
    #         transparent=False,
    #     )
    # else:
    #     plt.show()

    # # Now for the percentage of accounts reached
    # plt.figure()
    # for model in data:
    #     total_accounts = int(data[model]["simulation_data"]["population"])
    #     time = data[model]["time"]
    #     news_reached_per_id = data[model]["news_reached_per_id"]
    #     for news_id in news_ids:
    #         plt.plot(
    #             time,
    #             np.array(news_reached_per_id[news_id]) / total_accounts,
    #             label=f"{model}",
    #         )

    # # Set the axis
    # plt.xlabel("Time")
    # plt.xlim(0, max_time)
    # plt.ylabel("Accounts reached (%)")
    # plt.ylim(0, 1)
    # plt.grid()
    # if legend:
    #     plt.legend()
    # if graph_filename:
    #     plt.savefig(
    #         f"{graph_filename}_percentage_with_old.png",
    #         dpi=300,
    #         format="png",
    #         bbox_inches="tight",
    #         transparent=False,
    #     )
    # else:
    #     plt.show()

    # # Now for the cummulative number of accounts reached
    # plt.figure()
    # for model in data:
    #     total_accounts = int(data[model]["simulation_data"]["population"])
    #     time = data[model]["time"]
    #     news_reached_per_id = data[model]["news_reached_per_id"]
    #     for news_id in news_ids:
    #         plt.plot(
    #             time,
    #             np.cumsum(news_reached_per_id[news_id]) / total_accounts,
    #             label=f"{model}",
    #         )

    # # Set the axis
    # plt.xlabel("Time")
    # plt.xlim(0, max_time)
    # plt.ylabel("Cumulative number of accounts reached (%)")
    # plt.ylim(0, 1.01)
    # plt.grid()
    # if legend:
    #     plt.legend(loc="center right")

    # if graph_filename:
    #     plt.savefig(
    #         f"{graph_filename}_cumulative_with_old.png",
    #         dpi=300,
    #         format="png",
    #         bbox_inches="tight",
    #         transparent=False,
    #     )
    # else:
    #     plt.show()

    # Now x axis the number of accounts that have sent the news and y axis the number of accounts reached
    plt.figure()
    x = []
    y = []
    for model in data:
        total_accounts = int(data[model]["simulation_data"]["population"])
        news_reached_per_id = data[model]["news_reached_per_id"]
        for news_id in news_ids:
            x.append(int(model.split("(")[-1].split(")")[0]))
            y.append(int(sum(news_reached_per_id[news_id])))
    plt.plot(x, y, label=f"Final model")

    # Set the axis
    plt.xlabel("Number of fake accounts")
    plt.xlim(0, 400)
    plt.ylabel("Number of accounts reached")
    plt.ylim(0, max_population)
    plt.grid()
    if legend:
        plt.legend()

    if graph_filename:
        plt.savefig(
            f"{graph_filename}_totals.png",
            dpi=300,
            format="png",
            bbox_inches="tight",
            transparent=False,
        )


if __name__ == "__main__":
    version = "VFinal"
    pop = 40000
    accounts = [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
        160,
        170,
        180,
        190,
        200,
        210,
        220,
        230,
        240,
        250,
        260,
        270,
        280,
        290,
        300,
        310,
        320,
        330,
        340,
        350,
        360,
        370,
        380,
        390,
        400,
        410,
        420,
        430,
        440,
        450,
        460,
        470,
        480,
        490,
        500,
        510,
        520,
        530,
        540,
        550,
        560,
        570,
        580,
        590,
        600,
    ]

    version_pop_betas = []
    for num_accouts in accounts:
        version_pop_betas.append(
            f"C:/Users/larsv/OneDrive/Documenten/TU Delft/Vakken/Modelling 2b (AM2050-B)/Modelling-2B/tmp/{version}/{pop}/{num_accouts}/news_reached.csv"
        )

    version_pop_data = []
    for version_pop_beta in version_pop_betas:
        version_pop_data.append(get_simulation_data(os.path.dirname(version_pop_beta)))

    version_pop_names = []
    for num_accouts in accounts:
        version_pop_names.append(f"Model {version} ({num_accouts})")

    graph_filename = f"C:/Users/larsv/OneDrive/Documenten/TU Delft/Vakken/Modelling 2b (AM2050-B)/Modelling-2B/tmp/{version}/{pop}/news_reached_graph.png"

    plot_news_reached_3(
        csv_filenames=version_pop_betas,
        graph_filename=graph_filename,
        simulation_data=version_pop_data,
        model_names=version_pop_names,
    )
