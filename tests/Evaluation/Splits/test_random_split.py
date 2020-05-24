from DRecPy.Evaluation.Splits import random_split
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
    ], columns=['user', 'item', 'interaction', 'timestamp'])
    return InteractionDataset.read_df(df)


def test_random_split_0(interactions_ds):
    """Test if error is thrown with an invalid value of test_ratio (zero)."""
    try:
        random_split(interactions_ds, test_ratio=0)
    except Exception as e:
        assert str(e) == 'The test_ratio argument must be in the (0, 1) range.'


def test_random_split_1(interactions_ds):
    """Test if error is thrown with an invalid value of test_ratio (negative)."""
    try:
        random_split(interactions_ds, test_ratio=-1)
    except Exception as e:
        assert str(e) == 'The test_ratio argument must be in the (0, 1) range.'


def test_random_split_2(interactions_ds):
    """Test if error is thrown with an invalid value of test_ratio (1)."""
    try:
        random_split(interactions_ds, test_ratio=1)
    except Exception as e:
        assert str(e) == 'The test_ratio argument must be in the (0, 1) range.'


def test_random_split_3(interactions_ds):
    """Test if error is thrown with an invalid value of test_ratio (> 1)."""
    try:
        random_split(interactions_ds, test_ratio=2)
    except Exception as e:
        assert str(e) == 'The test_ratio argument must be in the (0, 1) range.'


def test_random_split_4(interactions_ds):
    """Test with half of the records sampled to the test set."""
    train_ds, test_ds = random_split(interactions_ds, test_ratio=0.5, seed=0)
    assert [[1, 2, 3, 100, 0], [1, 4, 5, 50, 1], [1, 5, 2, 25, 2]] == train_ds.values_list(to_list=True)
    assert [[2, 2, 5, 100, 3], [2, 3, 2, 20, 4]] == test_ds.values_list(to_list=True)


def test_random_split_5(interactions_ds):
    """Test with more than half of the records sampled to the test set."""
    train_ds, test_ds = random_split(interactions_ds, test_ratio=0.8, seed=0)
    assert [[1, 5, 2, 25, 2]] == train_ds.values_list(to_list=True)
    assert [[1, 2, 3, 100, 0], [1, 4, 5, 50, 1], [2, 2, 5, 100, 3], [2, 3, 2, 20, 4]] == test_ds.values_list(to_list=True)


def test_random_split_6(interactions_ds):
    """Test with less than half of the records sampled to the test set."""
    train_ds, test_ds = random_split(interactions_ds, test_ratio=0.2, seed=0)
    assert [[1, 2, 3, 100, 0], [1, 4, 5, 50, 1], [1, 5, 2, 25, 2], [2, 3, 2, 20, 4]] == train_ds.values_list(to_list=True)
    assert [[2, 2, 5, 100, 3]] == test_ds.values_list(to_list=True)