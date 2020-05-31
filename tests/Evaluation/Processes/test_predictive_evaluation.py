from DRecPy.Evaluation.Processes import predictive_evaluation
import pytest
import pandas as pd
from DRecPy.Dataset import InteractionDataset
from DRecPy.Recommender.Baseline import UserKNN
from DRecPy.Evaluation.Metrics import rmse
from DRecPy.Evaluation.Metrics import mse


@pytest.fixture(scope='module')
def train_interactions_ds():
    df = pd.DataFrame([
        [1, 2, 3],
        [1, 4, 5],
        [1, 5, 2],
        [2, 2, 5],
        [2, 3, 2],
        [3, 2, 2],
        [3, 5, 5],
        [3, 1, 1],
    ], columns=['user', 'item', 'interaction'])
    return InteractionDataset.read_df(df)


@pytest.fixture(scope='module')
def test_interactions_ds():
    df = pd.DataFrame([
        [1, 1, 2],
        [2, 4, 5],
        [3, 3, 3],
        [3, 6, 1],
    ], columns=['user', 'item', 'interaction'])
    return InteractionDataset.read_df(df)


@pytest.fixture(scope='module')
def model(train_interactions_ds):
    item_knn = UserKNN(k=3, m=0, sim_metric='cosine', aggregation='weighted_mean',
                       shrinkage=100, use_averages=False)
    item_knn.fit(train_interactions_ds, verbose=False)
    return item_knn


def test_predictive_evaluation_0(model, test_interactions_ds):
    """Evaluation without counting None predictions."""
    assert predictive_evaluation(model, test_interactions_ds, count_none_predictions=False,
                                 n_test_predictions=None, skip_errors=True) == {'MSE': 0.6667, 'RMSE': 0.8165}


def test_predictive_evaluation_1(model, test_interactions_ds):
    """Evaluation counting None predictions."""
    assert predictive_evaluation(model, test_interactions_ds, count_none_predictions=True,
                                 n_test_predictions=None, skip_errors=True) == {'MSE': 0.75, 'RMSE': 0.866}


def test_predictive_evaluation_2(model, test_interactions_ds):
    """Evaluation without skip errors."""
    try:
        predictive_evaluation(model, test_interactions_ds, count_none_predictions=False,
                              n_test_predictions=None, skip_errors=False) == {'MSE': 0.75, 'RMSE': 0.866}
        assert False
    except Exception as e:
        assert str(e) == 'Item 6 was not found.'


def test_predictive_evaluation_3(model, test_interactions_ds):
    """Evaluation without skip errors."""
    try:
        predictive_evaluation(model, test_interactions_ds, count_none_predictions=True,
                              n_test_predictions=None, skip_errors=False) == {'MSE': 0.75, 'RMSE': 0.866}
        assert False
    except Exception as e:
        assert str(e) == 'Item 6 was not found.'


def test_predictive_evaluation_4(model, test_interactions_ds):
    """Evaluation using the RMSE metric only."""
    assert predictive_evaluation(model, test_interactions_ds, count_none_predictions=False,
                                 n_test_predictions=None, skip_errors=True,
                                 metrics={'RMSE': (rmse, {})}) == {'RMSE': 0.8165}


def test_predictive_evaluation_5(model, test_interactions_ds):
    """Evaluation using the MSE metric only."""
    assert predictive_evaluation(model, test_interactions_ds, count_none_predictions=False,
                                 n_test_predictions=None, skip_errors=True,
                                 metrics={'MSE': (mse, {})}) == {'MSE': 0.6667}


def test_predictive_evaluation_6(model, test_interactions_ds):
    """Evaluation on the first test prediction."""
    assert predictive_evaluation(model, test_interactions_ds, count_none_predictions=False,
                                 n_test_predictions=1, skip_errors=True) == {'MSE': 1.0, 'RMSE': 1.0}


def test_predictive_evaluation_7(model, test_interactions_ds):
    """Evaluation on the first 2 test predictions."""
    assert predictive_evaluation(model, test_interactions_ds, count_none_predictions=False,
                                 n_test_predictions=2, skip_errors=True) == {'MSE': 0.5, 'RMSE': 0.7071}


def test_predictive_evaluation_8(model):
    """Evaluation on the training set."""
    assert predictive_evaluation(model, count_none_predictions=False,
                                 n_test_predictions=None, skip_errors=True) == {'MSE': 5.2485, 'RMSE': 2.291}


def test_predictive_evaluation_9(model, test_interactions_ds):
    """Invalid n_test_predictions value (0)."""
    try:
        predictive_evaluation(model, test_interactions_ds, count_none_predictions=False,
                              n_test_predictions=0, skip_errors=True)
        assert False
    except Exception as e:
        assert str(e) == 'The number of test users (0) should be > 0.'


def test_predictive_evaluation_10(model, test_interactions_ds):
    """Invalid n_test_predictions value (< 0)."""
    try:
        predictive_evaluation(model, test_interactions_ds, count_none_predictions=False,
                              n_test_predictions=-1, skip_errors=True)
        assert False
    except Exception as e:
        assert str(e) == 'The number of test users (-1) should be > 0.'


def test_predictive_evaluation_11(model, test_interactions_ds):
    """Invalid metrics value (not a dict)."""
    try:
        predictive_evaluation(model, test_interactions_ds, count_none_predictions=False,
                              n_test_predictions=None, skip_errors=True, metrics=[])
        assert False
    except Exception as e:
        assert str(e) == 'Expected "metrics" argument to be of type dict and found <class '"'list'"'>. ' \
                         'Should map metric names to a tuple containing the corresponding metric function and an ' \
                         'extra argument dict.'


def test_predictive_evaluation_12(model, test_interactions_ds):
    """Invalid metrics value (dict with non-tuple values)."""
    try:
        predictive_evaluation(model, test_interactions_ds, count_none_predictions=False,
                              n_test_predictions=None, skip_errors=True, metrics={'A': rmse})
        assert False
    except Exception as e:
        assert str(e) == 'Expected metric A to map to a tuple containing the corresponding metric function and an ' \
                         'extra argument dict.'


def test_predictive_evaluation_13(model, test_interactions_ds):
    """Invalid metrics value (dict with tuple values containing non-callables on the first element)."""
    try:
        predictive_evaluation(model, test_interactions_ds, count_none_predictions=False,
                              n_test_predictions=None, skip_errors=True, metrics={'A': (1, {})})
        assert False
    except Exception as e:
        assert str(e) == 'Expected metric A to map to a tuple containing the corresponding metric function and an ' \
                         'extra argument dict.'


def test_predictive_evaluation_14(model, test_interactions_ds):
    """Invalid metrics value (dict with tuple values containing non-dicts on the second element)."""
    try:
        predictive_evaluation(model, test_interactions_ds, count_none_predictions=False,
                              n_test_predictions=None, skip_errors=True, metrics={'A': (mse, [])})
        assert False
    except Exception as e:
        assert str(e) == 'Expected metric A to map to a tuple containing the corresponding metric function and an ' \
                         'extra argument dict.'
