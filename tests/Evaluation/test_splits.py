from DRecPy.Evaluation import leave_k_out
import pandas as pd
import numpy as np


""" leave_k_out """
def test_leave_k_out_0():
    """Test if error is thrown with an invalid value of k (zero)."""
    try:
        leave_k_out(pd.DataFrame(), 0)
    except Exception as e:
        assert str(e) == 'The value of k (0) must be > 0.'


def test_leave_k_out_1():
    """Test if error is thrown with an invalid value of k (negative)."""
    try:
        leave_k_out(pd.DataFrame(), -999)
    except Exception as e:
        assert str(e) == 'The value of k (-999) must be > 0.'


def test_leave_k_out_2():
    """Test if the return is correct by splitting k ratings from users with more than k ratings."""
    df = pd.DataFrame([
        [1, 2, 3],
        [1, 4, 5],
        [1, 5, 2],
        [2, 2, 5]
    ], columns=['user', 'item', 'rating'])
    df_train, df_test = leave_k_out(df, k=1, min_user_ratings=1)
    assert np.array_equal(df_train.values, [[2, 2, 5], [1, 2, 3], [1, 5, 2]])
    assert np.array_equal(df_test.values, [[1, 4, 5]])


def test_leave_k_out_3():
    """Test if the return is correct by not splitting if there are no users with k minimum item ratings."""
    df = pd.DataFrame([
        [1, 2, 3],
        [1, 4, 5],
        [1, 5, 2],
        [2, 2, 5]
    ], columns=['user', 'item', 'rating'])
    df_train, df_test = leave_k_out(df, k=3, min_user_ratings=1)
    assert np.array_equal(df_train.values, [[2, 2, 5], [1, 2, 3], [1, 5, 2], [1, 4, 5]])
    assert len(df_test.values) == 0


def test_leave_k_out_4():
    """Test if the return is correct by filtering out users without the minimum number of item ratings."""
    df = pd.DataFrame([
        [1, 2, 3],
        [1, 4, 5],
        [1, 5, 2],
        [2, 2, 5]
    ], columns=['user', 'item', 'rating'])
    df_train, df_test = leave_k_out(df, k=1, min_user_ratings=4)
    assert len(df_train.values) == 0
    assert len(df_test.values) == 0
