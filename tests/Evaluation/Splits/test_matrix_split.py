from DRecPy.Evaluation.Splits import matrix_split
import pandas as pd
import pytest
from DRecPy.Dataset import InteractionDataset


@pytest.fixture
def interactions_ds():
    df = pd.DataFrame([
        [1, 2, 3, 100],
        [1, 4, 5, 50],
        [1, 5, 2, 25],
        [2, 2, 5, 100],
        [2, 3, 2, 20],
        [2, 4, 2, 20],
        [3, 2, 1, 200],
        [3, 3, 5, 250],
    ], columns=['user', 'item', 'interaction', 'timestamp'])
    return InteractionDataset.read_df(df)


def test_matrix_split_0(interactions_ds):
    """Test if error is thrown with an invalid value of user_test_ratio (zero)."""
    try:
        matrix_split(interactions_ds, user_test_ratio=0)
    except Exception as e:
        assert str(e) == 'Invalid user_test_ratio of 0: must be in the range (0, 1]'


def test_matrix_split_1(interactions_ds):
    """Test if error is thrown with an invalid value of user_test_ratio (negative)."""
    try:
        matrix_split(interactions_ds, user_test_ratio=-1)
    except Exception as e:
        assert str(e) == 'Invalid user_test_ratio of -1: must be in the range (0, 1]'


def test_matrix_split_3(interactions_ds):
    """Test if error is thrown with an invalid value of user_test_ratio (> 1)."""
    try:
        matrix_split(interactions_ds, user_test_ratio=2)
    except Exception as e:
        assert str(e) == 'Invalid user_test_ratio of 2: must be in the range (0, 1]'


def test_matrix_split_4(interactions_ds):
    """Test if error is thrown with an invalid value of item_test_ratio (zero)."""
    try:
        matrix_split(interactions_ds, item_test_ratio=0)
    except Exception as e:
        assert str(e) == 'Invalid item_test_ratio of 0: must be in the range (0, 1]'


def test_matrix_split_5(interactions_ds):
    """Test if error is thrown with an invalid value of item_test_ratio (negative)."""
    try:
        matrix_split(interactions_ds, item_test_ratio=-1)
    except Exception as e:
        assert str(e) == 'Invalid item_test_ratio of -1: must be in the range (0, 1]'


def test_matrix_split_7(interactions_ds):
    """Test if error is thrown with an invalid value of item_test_ratio (> 1)."""
    try:
        matrix_split(interactions_ds, item_test_ratio=2)
    except Exception as e:
        assert str(e) == 'Invalid item_test_ratio of 2: must be in the range (0, 1]'


def test_matrix_split_8(interactions_ds):
    """Test if error is thrown with an invalid value of max_concurrent_threads (zero)."""
    try:
        matrix_split(interactions_ds, max_concurrent_threads=0)
    except Exception as e:
        assert str(e) == 'The value of max_concurrent_threads (0) must be > 0.'


def test_matrix_split_9(interactions_ds):
    """Test user_test_ratio = 0.5 and item_test_ratio = 0.5."""
    train_ds, test_ds = matrix_split(interactions_ds, user_test_ratio=0.5, item_test_ratio=0.5, seed=0)
    assert [[1, 2, 3, 100, 0], [1, 4, 5, 50, 1], [1, 5, 2, 25, 2], [2, 4, 2, 20, 5], [3, 2, 1, 200, 6], [3, 3, 5, 250, 7]] == train_ds.values_list(to_list=True)
    assert [[2, 2, 5, 100, 3], [2, 3, 2, 20, 4]] == test_ds.values_list(to_list=True)


def test_matrix_split_10(interactions_ds):
    """Test user_test_ratio = 0.7 and item_test_ratio = 0.5."""
    train_ds, test_ds = matrix_split(interactions_ds, user_test_ratio=0.7, item_test_ratio=0.5, seed=0)
    assert [[1, 2, 3, 100, 0], [1, 4, 5, 50, 1], [1, 5, 2, 25, 2], [2, 3, 2, 20, 4], [3, 3, 5, 250, 7]] == train_ds.values_list(to_list=True)
    assert [[2, 2, 5, 100, 3], [2, 4, 2, 20, 5], [3, 2, 1, 200, 6]] == test_ds.values_list(to_list=True)


def test_matrix_split_11(interactions_ds):
    """Test user_test_ratio = 0.7 and item_test_ratio = 0.75."""
    train_ds, test_ds = matrix_split(interactions_ds, user_test_ratio=0.7, item_test_ratio=0.75, seed=0)
    assert [[1, 2, 3, 100, 0], [1, 4, 5, 50, 1], [1, 5, 2, 25, 2], [2, 3, 2, 20, 4], [3, 3, 5, 250, 7]] == train_ds.values_list(to_list=True)
    assert [[2, 2, 5, 100, 3], [2, 4, 2, 20, 5], [3, 2, 1, 200, 6]] == test_ds.values_list(to_list=True)


def test_matrix_split_12(interactions_ds):
    """Test user_test_ratio = 0.75 and item_test_ratio = 0.75."""
    train_ds, test_ds = matrix_split(interactions_ds, user_test_ratio=0.75, item_test_ratio=0.75, seed=0)
    assert [[1, 2, 3, 100, 0], [1, 4, 5, 50, 1], [1, 5, 2, 25, 2], [2, 3, 2, 20, 4], [3, 3, 5, 250, 7]] == train_ds.values_list(to_list=True)
    assert [[2, 2, 5, 100, 3], [2, 4, 2, 20, 5], [3, 2, 1, 200, 6]] == test_ds.values_list(to_list=True)


def test_matrix_split_13(interactions_ds):
    """Test user_test_ratio = 1 and item_test_ratio = 0.75."""
    train_ds, test_ds = matrix_split(interactions_ds, user_test_ratio=1, item_test_ratio=0.75, seed=5)
    assert [[1, 5, 2, 25, 2], [2, 2, 5, 100, 3], [2, 3, 2, 20, 4], [2, 4, 2, 20, 5], [3, 2, 1, 200, 6], [3, 3, 5, 250, 7]] == train_ds.values_list(to_list=True)
    assert [[1, 2, 3, 100, 0], [1, 4, 5, 50, 1]] == test_ds.values_list(to_list=True)


def test_matrix_split_14(interactions_ds):
    """Test user_test_ratio = 1 and item_test_ratio = 0.2."""
    train_ds, test_ds = matrix_split(interactions_ds, user_test_ratio=1, item_test_ratio=0.2, seed=5)
    assert [[1, 2, 3, 100, 0], [1, 4, 5, 50, 1], [1, 5, 2, 25, 2], [2, 2, 5, 100, 3], [2, 3, 2, 20, 4], [2, 4, 2, 20, 5], [3, 2, 1, 200, 6], [3, 3, 5, 250, 7]] == train_ds.values_list(to_list=True)
    assert [] == test_ds.values_list(to_list=True)


def test_matrix_split_15(interactions_ds):
    """Test user_test_ratio = 0.2 and item_test_ratio = 1."""
    train_ds, test_ds = matrix_split(interactions_ds, user_test_ratio=0.2, item_test_ratio=1, seed=5)
    assert [[1, 2, 3, 100, 0], [1, 4, 5, 50, 1], [1, 5, 2, 25, 2], [2, 2, 5, 100, 3], [2, 3, 2, 20, 4], [2, 4, 2, 20, 5], [3, 2, 1, 200, 6], [3, 3, 5, 250, 7]] == train_ds.values_list(to_list=True)
    assert [] == test_ds.values_list(to_list=True)


def test_matrix_split_16(interactions_ds):
    """Test min_user_interactions = 3 so that users with #records < 3 should be removed from both sets."""
    train_ds, test_ds = matrix_split(interactions_ds, user_test_ratio=0.75, item_test_ratio=0.75, min_user_interactions=3, seed=0)
    assert [[1, 2, 3, 100, 0], [1, 4, 5, 50, 1], [1, 5, 2, 25, 2], [2, 3, 2, 20, 4]] == train_ds.values_list(to_list=True)
    assert [[2, 2, 5, 100, 3], [2, 4, 2, 20, 5]] == test_ds.values_list(to_list=True)