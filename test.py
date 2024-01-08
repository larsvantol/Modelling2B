import json

version = "VFinal"
pop = 40000

for sigma in list(range(10, 610, 10)):
    data = {"population": 40000, "version": "VFinal", "fake_accounts": sigma}

    path = f"C:/Users/larsv/OneDrive/Documenten/TU Delft/Vakken/Modelling 2b (AM2050-B)/Modelling-2B/tmp/{version}/{pop}/{sigma}/simulation.json"

    # Write the data to a json file
    with open(path, "w") as f:
        json.dump(data, f)
