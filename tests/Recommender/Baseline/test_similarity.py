from DRecPy.Recommender.Baseline.similarity import cosine_sim
from DRecPy.Recommender.Baseline.similarity import adjusted_cosine_sim
from DRecPy.Recommender.Baseline.similarity import cosine_sim_cf
from DRecPy.Recommender.Baseline.similarity import jaccard_sim
from DRecPy.Recommender.Baseline.similarity import pearson_corr
from DRecPy.Recommender.Baseline.similarity import msd
from DRecPy.Recommender.Baseline.similarity import _subtract_row_mean
from scipy.sparse import csr_matrix
import numpy as np
import pytest


@pytest.fixture
def input_array():
    return [[1, 2, 3], [1, 0, 0], [2, 4, 5], [3, 1, 5]]


@pytest.fixture
def input_matrix(input_array):
    return csr_matrix(input_array)


@pytest.fixture
def large_input_array():
    return [[4, 0, 0, 5, 1, 0, 0], [5, 5, 4, 0, 0, 0, 0], [0, 0, 0, 2, 4, 5, 0], [0, 3, 0, 0, 0, 0, 3]]


def test_cosine_sim_0(input_array):
    """Test if the correct similarity values are computed for array of arrays."""
    ans = [
        [1.    , 0.2673, 0.996 , 0.9035],
        [0.2673, 1.    , 0.2981, 0.5071],
        [0.996 , 0.2981, 1.    , 0.8819],
        [0.9035, 0.5071, 0.8819, 1.    ]
    ]
    ret = cosine_sim(input_array).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_cosine_sim_1(input_matrix):
    """Test if the correct similarity values are computed for csr_matrix types."""
    ans = [
        [1.    , 0.2673, 0.996 , 0.9035],
        [0.2673, 1.    , 0.2981, 0.5071],
        [0.996 , 0.2981, 1.    , 0.8819],
        [0.9035, 0.5071, 0.8819, 1.    ]
    ]
    ret = cosine_sim(input_matrix).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_cosine_sim_2(large_input_array):
    """Larger test for the correct similarity values."""
    ans = [
        [1.    , 0.3799, 0.322 , 0.    ],
        [0.3799, 1.    , 0.    , 0.4352],
        [0.322 , 0.    , 1.    , 0.    ],
        [0.    , 0.4352, 0.    , 1.    ]
    ]
    ret = cosine_sim(large_input_array).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_adjusted_cosine_sim_0(input_array):
    """Test if the correct similarity values are computed for array of arrays."""
    ans = [
        [1.    , 0.    , 0.982 , 0.5   ],
        [0.    , 0.    , 0.    , 0.    ],
        [0.982 , 0.    , 1.    , 0.3273],
        [0.5   , 0.    , 0.3273, 1.    ]
    ]
    ret = adjusted_cosine_sim(input_array).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_adjusted_cosine_sim_1(input_matrix):
    """Test if the correct similarity values are computed for csr_matrix types."""
    ans =  [
        [1.    , 0.    , 0.982 , 0.5   ],
        [0.    , 0.    , 0.    , 0.    ],
        [0.982 , 0.    , 1.    , 0.3273],
        [0.5   , 0.    , 0.3273, 1.    ]
    ]
    ret = adjusted_cosine_sim(input_matrix).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_adjusted_cosine_sim_2(large_input_array):
    """Larger test for the correct similarity values."""
    ans = [
        [ 1.    ,  0.0925, -0.5591,  0.    ],
        [ 0.0925,  1.    ,  0.    ,  0.    ],
        [-0.5591,  0.    ,  1.    ,  0.    ],
        [ 0.    ,  0.    ,  0.    ,  0.    ]
    ]
    ret = adjusted_cosine_sim(large_input_array).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_cosine_sim_cf_0(input_array):
    """Test if the correct similarity values are computed for array of arrays."""
    ans = [
        [1.    , 1.    , 0.996 , 0.9035],
        [1.    , 1.    , 1.    , 1.    ],
        [0.996 , 1.    , 1.    , 0.8819],
        [0.9035, 1.    , 0.8819, 1.    ]
    ]
    ret = cosine_sim_cf(input_array).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_cosine_sim_cf_1(input_matrix):
    """Test if the correct similarity values are computed for csr_matrix types."""
    ans = [
        [1.    , 1.    , 0.996 , 0.9035],
        [1.    , 1.    , 1.    , 1.    ],
        [0.996 , 1.    , 1.    , 0.8819],
        [0.9035, 1.    , 0.8819, 1.    ]
    ]
    ret = cosine_sim_cf(input_matrix).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_cosine_sim_cf_2(large_input_array):
    """Larger test for the correct similarity values."""
    ans = [
        [1.    , 1.    , 0.6139, 0.    ],
        [1.    , 1.    , 0.    , 1.    ],
        [0.6139, 0.    , 1.    , 0.    ],
        [0.    , 1.    , 0.    , 1.    ]
    ]
    ret = cosine_sim_cf(large_input_array).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_jaccard_sim_0(input_array):
    """Test if the correct similarity values are computed for array of arrays."""
    ans = [
        [1.    , 0.3333, 1.    , 1.    ],
        [0.3333, 1.    , 0.3333, 0.3333],
        [1.    , 0.3333, 1.    , 1.    ],
        [1.    , 0.3333, 1.    , 1.    ]
    ]
    ret = jaccard_sim(input_array).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_jaccard_sim_1(input_matrix):
    """Test if the correct similarity values are computed for csr_matrix types."""
    ans = [
        [1.    , 0.3333, 1.    , 1.    ],
        [0.3333, 1.    , 0.3333, 0.3333],
        [1.    , 0.3333, 1.    , 1.    ],
        [1.    , 0.3333, 1.    , 1.    ]
    ]
    ret = jaccard_sim(input_matrix).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_jaccard_sim_2(large_input_array):
    """Larger test for the correct similarity values."""
    ans = [
        [1.  , 0.2 , 0.5 , 0.  ],
        [0.2 , 1.  , 0.  , 0.25],
        [0.5 , 0.  , 1.  , 0.  ],
        [0.  , 0.25, 0.  , 1.  ]
    ]
    ret = jaccard_sim(large_input_array).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_pearson_corr_0(input_array):
    """Test if the correct similarity values are computed for array of arrays."""
    ans = [
        [1.    , 0.    , 0.982 , 0.5   ],
        [0.    , 0.    , 0.    , 0.    ],
        [0.982 , 0.    , 1.    , 0.3273],
        [0.5   , 0.    , 0.3273, 1.    ]
    ]
    ret = pearson_corr(input_array).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_pearson_corr_1(input_matrix):
    """Test if the correct similarity values are computed for csr_matrix types."""
    ans = [
        [1.    , 0.    , 0.982 , 0.5   ],
        [0.    , 0.    , 0.    , 0.    ],
        [0.982 , 0.    , 1.    , 0.3273],
        [0.5   , 0.    , 0.3273, 1.    ]
    ]
    ret = pearson_corr(input_matrix).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_pearson_corr_2(large_input_array):
    """Larger test for the correct similarity values."""
    ans = [
        [ 1.,  0., -1.],
        [ 0.,  1.,  0.],
        [-1.,  0.,  1.]
    ]
    ret = pearson_corr(large_input_array).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_msd_0(input_array):
    """Test if the correct similarity values are computed for array of arrays."""
    ans = [
        [1.    , 1.    , 0.88  , 0.88  ],
        [1.    , 1.    , 0.96  , 0.84  ],
        [0.88  , 0.96  , 1.    , 0.8667],
        [0.88  , 0.84  , 0.8667, 1.    ]
    ]
    ret = msd(input_array).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_msd_1(input_matrix):
    """Test if the correct similarity values are computed for csr_matrix types."""
    ans = [
        [1.    , 1.    , 0.88  , 0.88  ],
        [1.    , 1.    , 0.96  , 0.84  ],
        [0.88  , 0.96  , 1.    , 0.8667],
        [0.88  , 0.84  , 0.8667, 1.    ]
    ]
    ret = msd(input_matrix).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_msd_2(large_input_array):
    """Larger test for the correct similarity values."""
    ans = [
        [1.  , 0.96, 0.64, 0.  ],
        [0.96, 1.  , 0.  , 0.84],
        [0.64, 0.  , 1.  , 0.  ],
        [0.  , 0.84, 0.  , 1.  ]
    ]
    ret = msd(large_input_array).toarray()
    assert np.array_equal(np.around(ret, decimals=4), np.around(ans, decimals=4))


def test_subtract_row_mean_0():
    """Test if error is thrown when a non csr_matrix is given as parameter."""
    m = [[1, 2, 3], [1, 0, 0], [2, 4, 6]]
    try:
        _subtract_row_mean(m)
    except Exception as e:
        assert str(e) == 'The given argument should be of type scipy.sparse.csr_matrix'


def test_subtract_row_mean_1():
    """Test if the return value is correct."""
    m = csr_matrix([[1, 2, 3], [1, 0, 0], [2, 4, 6]])
    ans = [[-1, 0, 1], [0, 0, 0], [-2, 0, 2]]
    ret = _subtract_row_mean(m).toarray()
    assert np.array_equal(ans, ret)
