import json
import os


def get_simulation_data(folder, file_name="simulation.json"):
    filepath = os.path.join(folder, file_name)
    with open(filepath, "r") as file:
        simulation_data = json.load(file)
    return simulation_data
