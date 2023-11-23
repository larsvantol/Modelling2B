from tkinter.filedialog import askopenfilename

import numpy as np
import numpy.typing as npt


def open_node_information_array(
    input_file: str,
) -> npt.NDArray[np.float_]:
    """
    Open the node information from a file.
    """

    scores = np.loadtxt(input_file, dtype=float)

    return scores


if __name__ == "__main__":
    file = askopenfilename(
        filetypes=[("CSV", "*.csv")],
        title="Open adjacency matrix",
    )
    adjacency_matrix = open_node_information_array(file)
    print(adjacency_matrix)
