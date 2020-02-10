from scipy.sparse import csr_matrix
from scipy.sparse import diags
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def cosine_sim(matrix):
    """Given a matrix NxM, returns a new matrix with size NxN containing all the cosine
    similarities between each row to every other row of the provided matrix."""
    if type(matrix) is not csr_matrix:
        matrix = csr_matrix(matrix)

    return cosine_similarity(matrix, dense_output=False)


def adjusted_cosine_sim(matrix):
    """Given a matrix NxM, returns a new matrix with size NxN containing all the adjusted
    cosine similarities between each row to every other row of the provided matrix.
    Adjusted cosine sim. is the cosine sim. applied after subtracting the matrix column
    averages of each column values."""
    if type(matrix) is not csr_matrix:
        matrix = csr_matrix(matrix)

    matrix = subtract_row_mean(matrix)
    return cosine_similarity(matrix, dense_output=False)


def subtract_row_mean(A):
    """For each row in the given matrix, subtracts the row mean from every value in the row.

    Args:
        A: csr_matrix.

    Returns:
        A row-modified csr_matrix.
    """
    assert type(A) is csr_matrix, "The given argument should be of type scipy.sparse.csr_matrix"

    sum_rows = np.array(A.sum(axis=1).squeeze())[0]
    size_rows = np.diff(A.indptr)
    avg_rows = np.divide(sum_rows, size_rows, where=size_rows != 0)
    avg_diag_matrix = diags(avg_rows, 0)
    ones_matrix = A.copy()
    ones_matrix.data = np.ones_like(A.data)

    return A - avg_diag_matrix * ones_matrix
