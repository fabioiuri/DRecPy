from DRecPy.Similarity import get_sim_metric
from DRecPy.Similarity import cosine_sim
from DRecPy.Similarity import adjusted_cosine_sim
from scipy.sparse import csr_matrix
import numpy as np


""" get_sim_metric """
def test_get_sim_metric_0():
    """Test if error is thrown when metric with provided name is not found."""
    try:
        get_sim_metric('')
    except Exception as e:
        assert str(e) == 'There is no similarity metric corresponding to the name "".'


def test_get_sim_metric_1():
    """Test if cosine metric is returned."""
    assert get_sim_metric('cosine') == cosine_sim


def test_get_sim_metric_2():
    """Test if adjusted cosine metric is returned."""
    assert get_sim_metric('adjusted_cosine') == adjusted_cosine_sim


""" cosine_sim """
def test_cosine_sim_0():
    """Test if the correct similarity values are computed for array of arrays."""
    m = [[1, 2, 3], [1, 0, 0], [2, 4, 6]]
    ans = [
        [1, 0.2672612419124244, 1],
        [0.2672612419124244, 1, 0.2672612419124244],
        [1, 0.2672612419124244, 1]
    ]
    ret = cosine_sim(m).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_cosine_sim_1():
    """Test if the correct similarity values are computed for csr_matrix types."""
    m = csr_matrix([[1, 2, 3], [1, 0, 0], [2, 4, 6]])
    ans = [
        [1, 0.2672612419124244, 1],
        [0.2672612419124244, 1, 0.2672612419124244],
        [1, 0.2672612419124244, 1]
    ]
    ret = cosine_sim(m).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_cosine_sim_2():
    """Larger test for the correct similarity values."""
    m = [[4, 0, 0, 5, 1, 0, 0], [5, 5, 4, 0, 0, 0, 0], [0, 0, 0, 2, 4, 5, 0], [0, 3, 0, 0, 0, 0, 3]]
    ans = [
        [1, 0.37986859, 0.32203059, 0],
        [0.37986859, 1, 0, 0.43519414],
        [0.32203059, 0, 1, 0],
        [0, 0.43519414, 0, 1]
    ]
    ret = cosine_sim(m).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


""" adjusted_cosine_sim """
def test_adjusted_cosine_sim_0():
    """Test if the correct similarity values are computed for array of arrays."""
    m = [[1, 2, 3], [1, 0, 0], [2, 4, 6]]
    ans = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
    ret = adjusted_cosine_sim(m).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_adjusted_cosine_sim_1():
    """Test if the correct similarity values are computed for csr_matrix types."""
    m = csr_matrix([[1, 2, 3], [1, 0, 0], [2, 4, 6]])
    ans = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
    ret = adjusted_cosine_sim(m).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_adjusted_cosine_sim_2():
    """Larger test for the correct similarity values."""
    m = [[4, 0, 0, 5, 1, 0, 0], [5, 5, 4, 0, 0, 0, 0], [0, 0, 0, 2, 4, 5, 0], [0, 3, 0, 0, 0, 0, 3]]
    ans = [
        [1, 0.09245003, -0.55908525,  0],
        [0.09245003, 1, 0, 0],
        [-0.55908525,  0, 1, 0],
        [0, 0, 0, 0]
    ]
    ret = adjusted_cosine_sim(m).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))

