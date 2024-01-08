import math
from pathlib import Path
from tkinter.filedialog import askopenfilename

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm


def read_csv_with_progress(
    input_file: str,
) -> pd.DataFrame:
    """
    Open the adjacency matrix from a file.
    """

    txt = Path(input_file).resolve()

    print(f"Reading {txt}")
    # read number of rows quickly
    length = sum(1 for row in open(txt, "r"))

    # define a chunksize
    chunksize = 5000

    # initiate a blank dataframe
    df = pd.DataFrame()
    print(f"Reading {txt} in chunks of {chunksize} rows")
    # tqdm context
    with tqdm(total=length, desc="chunks read: ") as bar:
        # enumerate chunks read without low_memory (it is massive for pandas to precisely assign dtypes)
        for i, chunk in enumerate(
            pd.read_csv(txt, chunksize=chunksize, low_memory=False, header=None, delimiter=" ")
        ):
            # append it to df
            df = pd.concat([df, chunk])

            # update tqdm progress bar
            bar.update(chunksize)

    return df


def open_adjacency_matrix(
    input_file: str,
) -> npt.NDArray[np.bool_]:
    """
    Open the adjacency matrix from a file.
    """

    adjacency_matrix = read_csv_with_progress(input_file).to_numpy().astype(bool)

    return adjacency_matrix


if __name__ == "__main__":
    file = askopenfilename(
        filetypes=[("CSV", "*.csv")],
        title="Open adjacency matrix",
    )
    adjacency_matrix = open_adjacency_matrix(file)
    print(adjacency_matrix)
