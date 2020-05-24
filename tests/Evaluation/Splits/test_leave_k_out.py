from DRecPy.Evaluation.Splits import leave_k_out
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


@pytest.fixture
def interactions_ds_timestamp_label():
    df = pd.DataFrame([
        [1, 2, 3, 100],
        [1, 4, 5, 50],
        [1, 5, 2, 25],
        [2, 2, 5, 100],
        [2, 3, 2, 20],
    ], columns=['user', 'item', 'interaction', 'custom_timestamp_label'])
    return InteractionDataset.read_df(df)


def test_leave_k_out_0(interactions_ds):
    """Test if error is thrown with an invalid value of k (zero)."""
    try:
        leave_k_out(interactions_ds, 0)
    except Exception as e:
        assert str(e) == 'The value of k (0) must be > 0.'


def test_leave_k_out_1(interactions_ds):
    """Test if error is thrown with an invalid value of k (negative)."""
    try:
        leave_k_out(interactions_ds, -999)
    except Exception as e:
        assert str(e) == 'The value of k (-999) must be > 0.'


def test_leave_k_out_2(interactions_ds):
    """Test if error is thrown with an invalid value of k (ratio variant with negative k)."""
    try:
        leave_k_out(interactions_ds, -0.5)
    except Exception as e:
        assert str(e) == 'The value of k (-0.5) must be > 0.'


def test_leave_k_out_3(interactions_ds):
    """Test if error is thrown with an invalid value of k (ratio variant with k higher than 1)."""
    try:
        leave_k_out(interactions_ds, 1.5)
    except Exception as e:
        assert str(e) == 'The k parameter should be in the (0, 1) range when it\'s used as the percentage of ' \
                         'interactions to sample to the test set, per user. Current value: 1.5'


def test_leave_k_out_4(interactions_ds):
    """Test if error is thrown with an invalid value of max_concurrent_threads (zero)."""
    try:
        leave_k_out(interactions_ds, max_concurrent_threads=0)
    except Exception as e:
        assert str(e) == 'The value of max_concurrent_threads (0) must be > 0.'


def test_leave_k_out_5(interactions_ds):
    """Test if error is thrown with an invalid value of max_concurrent_threads (negative)."""
    try:
        leave_k_out(interactions_ds, max_concurrent_threads=-1)
    except Exception as e:
        assert str(e) == 'The value of max_concurrent_threads (-1) must be > 0.'


def test_leave_k_out_6(interactions_ds):
    """Test fixed k variant with value of k = 1."""
    train_ds, test_ds = leave_k_out(interactions_ds, k=1, seed=0)
    assert [[1, 4, 5, 50, 1], [1, 5, 2, 25, 2], [2, 3, 2, 20, 4]] == train_ds.values_list(to_list=True)
    assert [[1, 2, 3, 100, 0], [2, 2, 5, 100, 3]] == test_ds.values_list(to_list=True)


def test_leave_k_out_7(interactions_ds):
    """Test fixed k variant with value of k > 1. Should ignore users where #items <= k."""
    train_ds, test_ds = leave_k_out(interactions_ds, k=2, seed=0)
    assert [[1, 4, 5, 50, 1], [2, 2, 5, 100, 3], [2, 3, 2, 20, 4]] == train_ds.values_list(to_list=True)
    assert [[1, 2, 3, 100, 0], [1, 5, 2, 25, 2]] == test_ds.values_list(to_list=True)


def test_leave_k_out_8(interactions_ds):
    """Test fixed k variant with value of k > #available items per user."""
    train_ds, test_ds = leave_k_out(interactions_ds, k=100, seed=0)
    assert [[1, 2, 3, 100, 0], [1, 4, 5, 50, 1], [1, 5, 2, 25, 2], [2, 2, 5, 100, 3], [2, 3, 2, 20, 4]] == train_ds.values_list(to_list=True)
    assert [] == test_ds.values_list(to_list=True)


def test_leave_k_out_9(interactions_ds):
    """Test fixed k variant with min_user_interactions > 1. Should remove users from train and test sets that don't have at least min_user_interactions records."""
    train_ds, test_ds = leave_k_out(interactions_ds, k=1, min_user_interactions=3, seed=0)
    assert [[1, 4, 5, 50, 1], [1, 5, 2, 25, 2]] == train_ds.values_list(to_list=True)
    assert [[1, 2, 3, 100, 0]] == test_ds.values_list(to_list=True)


def test_leave_k_out_10(interactions_ds):
    """Test fixed k variant with last_timestamps = True."""
    train_ds, test_ds = leave_k_out(interactions_ds, k=1, last_timestamps=True, seed=0)
    assert [[1, 2, 3, 100, 0], [1, 4, 5, 50, 1], [2, 2, 5, 100, 3]] == train_ds.values_list(to_list=True)
    assert [[1, 5, 2, 25, 2], [2, 3, 2, 20, 4]] == test_ds.values_list(to_list=True)


def test_leave_k_out_11(interactions_ds_timestamp_label):
    """Test fixed k variant with last_timestamps = True with custom timestamp label."""
    train_ds, test_ds = leave_k_out(interactions_ds_timestamp_label, k=1, last_timestamps=True, timestamp_label='custom_timestamp_label', seed=0)
    assert [[1, 2, 3, 100, 0], [1, 4, 5, 50, 1], [2, 2, 5, 100, 3]] == train_ds.values_list(to_list=True)
    assert [[1, 5, 2, 25, 2], [2, 3, 2, 20, 4]] == test_ds.values_list(to_list=True)


def test_leave_k_out_12(interactions_ds):
    """Test ratio k variant with value of k resulting on sampled records for all users."""
    train_ds, test_ds = leave_k_out(interactions_ds, k=0.5, seed=0)
    assert [[1, 4, 5, 50, 1], [1, 5, 2, 25, 2], [2, 3, 2, 20, 4]] == train_ds.values_list(to_list=True)
    assert [[1, 2, 3, 100, 0], [2, 2, 5, 100, 3]] == test_ds.values_list(to_list=True)


def test_leave_k_out_13(interactions_ds):
    """Test ratio k variant with value of k resulting on sampled records for some but not all users."""
    train_ds, test_ds = leave_k_out(interactions_ds, k=0.34, seed=0)
    assert [[1, 4, 5, 50, 1], [1, 5, 2, 25, 2], [2, 2, 5, 100, 3], [2, 3, 2, 20, 4]] == train_ds.values_list(to_list=True)
    assert [[1, 2, 3, 100, 0]] == test_ds.values_list(to_list=True)


def test_leave_k_out_14(interactions_ds):
    """Test ratio k variant with value of k resulting on no sampled records."""
    train_ds, test_ds = leave_k_out(interactions_ds, k=0.3, seed=0)
    assert [[1, 2, 3, 100, 0], [1, 4, 5, 50, 1], [1, 5, 2, 25, 2], [2, 2, 5, 100, 3], [2, 3, 2, 20, 4]] == train_ds.values_list(to_list=True)
    assert [] == test_ds.values_list(to_list=True)


def test_leave_k_out_15(interactions_ds):
    """Test ratio k variant with min_user_interactions > 1. Should remove users from train and test sets that don't have at least min_user_interactions records."""
    train_ds, test_ds = leave_k_out(interactions_ds, k=0.4, min_user_interactions=3, seed=0)
    assert [[1, 4, 5, 50, 1], [1, 5, 2, 25, 2]] == train_ds.values_list(to_list=True)
    assert [[1, 2, 3, 100, 0]] == test_ds.values_list(to_list=True)


def test_leave_k_out_16(interactions_ds):
    """Test ratio k variant with last_timestamps = True."""
    train_ds, test_ds = leave_k_out(interactions_ds, k=0.5, last_timestamps=True, seed=0)
    assert [[1, 2, 3, 100, 0], [1, 4, 5, 50, 1], [2, 2, 5, 100, 3]] == train_ds.values_list(to_list=True)
    assert [[1, 5, 2, 25, 2], [2, 3, 2, 20, 4]] == test_ds.values_list(to_list=True)


def test_leave_k_out_17(interactions_ds_timestamp_label):
    """Test fixed k variant with last_timestamps = True with custom timestamp label."""
    train_ds, test_ds = leave_k_out(interactions_ds_timestamp_label, k=0.5, last_timestamps=True, timestamp_label='custom_timestamp_label', seed=0)
    assert [[1, 2, 3, 100, 0], [1, 4, 5, 50, 1], [2, 2, 5, 100, 3]] == train_ds.values_list(to_list=True)
    assert [[1, 5, 2, 25, 2], [2, 3, 2, 20, 4]] == test_ds.values_list(to_list=True)
