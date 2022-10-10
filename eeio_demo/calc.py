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


def scale_diagonal_to_unit(matrix: npt.ArrayLike) -> np.ndarray:
    """
    Scale the columns, \\(j\\), of the matrix, \\(A\\),
    to 1 unit of product \\(j\\) or -1 if it
    is a waste flow. So that, \\(A[j,j] = 1 (or -1)\\)

    Args:
        matrix (npt.ArrayLike): Matrix to be scaled.

    Returns:
        np.ndarray: Scaled matrix
    """
    matrix_in = np.array(matrix)
    scaled_matrix = np.copy(matrix)

    for i in range(scaled_matrix.shape[0]):
        scaled_matrix[:, i] /= matrix_in[i, i]

    return scaled_matrix
