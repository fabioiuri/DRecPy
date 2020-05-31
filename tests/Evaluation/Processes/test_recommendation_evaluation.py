from DRecPy.Evaluation.Processes import recommendation_evaluation
from DRecPy.Evaluation.Splits import leave_k_out
import pytest
import pandas as pd
from DRecPy.Dataset import InteractionDataset
from DRecPy.Recommender.Baseline import UserKNN
from DRecPy.Evaluation.Metrics import hit_ratio
from DRecPy.Evaluation.Metrics import ndcg
import random


@pytest.fixture(scope='module')
def interactions_ds():
    rng = random.Random(0)
    df = pd.DataFrame([
        [u, i, rng.randint(-1, 5)] for u in range(50) for i in range(200) if rng.randint(0, 4) == 0
    ], columns=['user', 'item', 'interaction'])
    print(df.values)
    return leave_k_out(InteractionDataset.read_df(df), k=5, min_user_interactions=0, last_timestamps=False, seed=10)


@pytest.fixture(scope='module')
def model(interactions_ds):
    item_knn = UserKNN(k=3, m=0, sim_metric='cosine', aggregation='weighted_mean',
                       shrinkage=100, use_averages=False)
    item_knn.fit(interactions_ds[0], verbose=False)
    return item_knn


def test_recommendation_evaluation_0(model, interactions_ds):
    assert recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=2, n_pos_interactions=None,
                                     novelty=False) == \
           {'AP@2': 0.01, 'HR@2': 0.01, 'NDCG@2': 0.0189, 'P@2': 0.02, 'R@2': 0.01, 'RR@2': 0.0}


def test_recommendation_evaluation_1(model, interactions_ds):
    """Evaluation with k parameter set to a list."""
    assert recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=[1, 5, 10],
                                     n_pos_interactions=None, novelty=False, verbose=False) == \
           {'AP@1': 0.0, 'AP@10': 0.0126, 'AP@5': 0.009, 'HR@1': 0.0, 'HR@10': 0.0457, 'HR@5': 0.0247,
            'NDCG@1': -0.0003, 'NDCG@10': 0.0329, 'NDCG@5': 0.0223, 'P@1': 0.0, 'P@10': 0.02, 'P@5': 0.02, 'R@1': 0.0,
            'R@10': 0.0457, 'R@5': 0.0247, 'RR@1': 0.0, 'RR@10': 0.0075, 'RR@5': 0.005}


def test_recommendation_evaluation_2(model, interactions_ds):
    """Evaluation with novelty=True."""
    assert recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=2, n_pos_interactions=None,
                                     novelty=True) == \
           {'AP@2': 0.025, 'HR@2': 0.0207, 'NDCG@2': 0.0263, 'P@2': 0.04, 'R@2': 0.0207, 'RR@2': 0.01}


def test_recommendation_evaluation_3(model, interactions_ds):
    """Evaluation with limited number of positive interactions."""
    assert recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=2, n_pos_interactions=1,
                                     novelty=False) == \
           {'AP@2': 0.01, 'HR@2': 0.02, 'NDCG@2': 0.0308, 'P@2': 0.01, 'R@2': 0.02, 'RR@2': 0.01}


def test_recommendation_evaluation_4(model, interactions_ds):
    """Evaluation with limited number of test users."""
    assert recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=2, n_pos_interactions=None,
                                     novelty=False, n_test_users=10) \
           == {'AP@2': 0.025, 'HR@2': 0.025, 'NDCG@2': 0.0585, 'P@2': 0.05, 'R@2': 0.025, 'RR@2': 0.0}


def test_recommendation_evaluation_5(model):
    """Train evaluation."""
    assert recommendation_evaluation(model, cn_test_users=None, k=2, n_pos_interactions=None, novelty=False) == \
           {'AP@2': 0.23, 'HR@2': 0.0164, 'NDCG@2': 0.0952, 'P@2': 0.25, 'R@2': 0.0164, 'RR@2': 0.02}


def test_recommendation_evaluation_6(model):
    """Train evaluation with novelty=True should result in all 0s."""
    assert recommendation_evaluation(model, cn_test_users=None, k=2, n_pos_interactions=None, novelty=True) == \
           {'AP@2': 0.0, 'HR@2': 0.0, 'NDCG@2': 0.0, 'P@2': 0, 'R@2': 0.0, 'RR@2': 0.0}


def test_recommendation_evaluation_7(model, interactions_ds):
    """Evaluation with a custom interaction threshold."""
    assert recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=2, n_pos_interactions=None,
                                     novelty=False, interaction_threshold=2) == \
           {'AP@2': 0.0052, 'HR@2': 0.0069, 'NDCG@2': 0.0116, 'P@2': 0.0104, 'R@2': 0.0069, 'RR@2': 0.0}


def test_recommendation_evaluation_8(model, interactions_ds):
    """Evaluation with custom metrics."""
    assert recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=2, n_pos_interactions=None,
                                     novelty=False, metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})}) == \
           {'HR@2': 0.01, 'NDCG@2': 0.0189}


def test_recommendation_evaluation_9(model, interactions_ds):
    """Evaluation with custom metrics and k set to a list."""
    assert recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=[2, 3], n_pos_interactions=None,
                                     novelty=False, metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})},
                                     verbose=False) == \
           {'HR@2': 0.01, 'HR@3': 0.0167, 'NDCG@2': 0.0189, 'NDCG@3': 0.022}


def test_recommendation_evaluation_10(model, interactions_ds):
    """Evaluation with invalid number of test users (0)."""
    try:
        recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=[2, 3], n_pos_interactions=None,
                                  novelty=False, n_test_users=0, metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})},
                                  verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of test users (0) should be > 0.'


def test_recommendation_evaluation_11(model, interactions_ds):
    """Evaluation with invalid number of test users (< 0)."""
    try:
        recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=[1, 2], n_pos_interactions=None,
                                  novelty=False, n_test_users=-1,  metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})},
                                  verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of test users (-1) should be > 0.'


def test_recommendation_evaluation_12(model, interactions_ds):
    """Evaluation with invalid number of positive interactions (0)."""
    try:
        recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=[1, 2], n_pos_interactions=0,
                                  novelty=False, metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of positive interactions (0) should be None or an integer > 0.'


def test_recommendation_evaluation_13(model, interactions_ds):
    """Evaluation with invalid number of positive interactions (< 0)."""
    try:
        recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=[1, 2], n_pos_interactions=-1,
                                  novelty=False, metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of positive interactions (-1) should be None or an integer > 0.'


def test_recommendation_evaluation_14(model, interactions_ds):
    """Evaluation with invalid number of k (0)."""
    try:
        recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=0, n_pos_interactions=None,
                                  novelty=False, metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'k (0) should be > 0.'


def test_recommendation_evaluation_15(model, interactions_ds):
    """Evaluation with invalid number of k (< 0)."""
    try:
        recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=-1, n_pos_interactions=None,
                                  novelty=False, metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'k (-1) should be > 0.'


def test_recommendation_evaluation_16(model, interactions_ds):
    """Invalid metrics value (not a dict)."""
    try:
        recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=5, n_pos_interactions=None,
                                  novelty=False, metrics=[], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'Expected "metrics" argument to be of type dict and found <class '"'list'"'>. ' \
                         'Should map metric names to a tuple containing the corresponding metric function and an ' \
                         'extra argument dict.'


def test_recommendation_evaluation_17(model, interactions_ds):
    """Invalid metrics value (dict with non-tuple values)."""
    try:
        recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=5, n_pos_interactions=None,
                                  novelty=False, metrics={'A': ndcg}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'Expected metric A to map to a tuple containing the corresponding metric function and an ' \
                         'extra argument dict.'


def test_recommendation_evaluation_18(model, interactions_ds):
    """Invalid metrics value (dict with tuple values containing non-callables on the first element)."""
    try:
        recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=5, n_pos_interactions=None,
                                  novelty=False, metrics={'A': (1, {})}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'Expected metric A to map to a tuple containing the corresponding metric function and an ' \
                         'extra argument dict.'


def test_recommendation_evaluation_19(model, interactions_ds):
    """Invalid metrics value (dict with tuple values containing non-dicts on the second element)."""
    try:
        recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=5, n_pos_interactions=None,
                                  novelty=False, metrics={'A': (ndcg, [])}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'Expected metric A to map to a tuple containing the corresponding metric function and an ' \
                         'extra argument dict.'
