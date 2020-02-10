from DRecPy.Similarity.utils import subtract_row_mean
from scipy.sparse import csr_matrix
import numpy as np


""" subtract_row_mean """
def test_subtract_row_mean_0():
    """Test if error is thrown when a non csr_matrix is given as parameter."""
    m = [[1, 2, 3], [1, 0, 0], [2, 4, 6]]
    try:
        subtract_row_mean(m)
    except Exception as e:
        assert str(e) == 'The given argument should be of type scipy.sparse.csr_matrix'


def test_subtract_row_mean_1():
    """Test if the return value is correct."""
    m = csr_matrix([[1, 2, 3], [1, 0, 0], [2, 4, 6]])
    ans = [[-1, 0, 1], [0, 0, 0], [-2, 0, 2]]
    ret = subtract_row_mean(m).toarray()
    assert np.array_equal(ans, ret)
