from tkinter.filedialog import askopenfilename

import numpy as np
import numpy.typing as npt


def open_adjacency_matrix(
    input_file: str,
) -> npt.NDArray[np.bool_]:
    """
    Open the adjacency matrix from a file.
    """

    adjacency_matrix = np.loadtxt(input_file, dtype=bool)

    return adjacency_matrix


if __name__ == "__main__":
    file = askopenfilename(
        filetypes=[("CSV", "*.csv")],
        title="Open adjacency matrix",
    )
    adjacency_matrix = open_adjacency_matrix(file)
    print(adjacency_matrix)
