import numpy as np
import numpy.typing as npt


def inv(matrix: npt.ArrayLike) -> np.ndarray:
    """
    Compute the inverse of a matrix.

    Args:
        matrix (npt.ArrayLike): Matrix to be inverted.

    Returns:
        np.ndarray: Inverted matrix
    """
    return np.linalg.inv(np.array(matrix))
