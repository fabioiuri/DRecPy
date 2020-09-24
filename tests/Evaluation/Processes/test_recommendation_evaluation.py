from DRecPy.Evaluation.Processes import recommendation_evaluation
from DRecPy.Evaluation.Splits import leave_k_out
import pytest
import pandas as pd
from DRecPy.Dataset import InteractionDataset
from DRecPy.Recommender.Baseline import UserKNN
from DRecPy.Evaluation.Metrics import HitRatio
from DRecPy.Evaluation.Metrics import NDCG
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
           {'HitRatio@2': 0.0167, 'NDCG@2': 0.0189, 'Precision@2': 0.02, 'Recall@2': 0.0167}


def test_recommendation_evaluation_1(model, interactions_ds):
    """Evaluation with k parameter set to a list."""
    assert recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=[1, 5, 10],
                                     n_pos_interactions=None, novelty=False, verbose=False) == \
           {'HitRatio@1': 0.0, 'HitRatio@10': 0.0507, 'HitRatio@5': 0.0283, 'NDCG@1': -0.0003, 'NDCG@10': 0.0329,
            'NDCG@5': 0.0223, 'Precision@1': 0.0, 'Precision@10': 0.016, 'Precision@5': 0.016, 'Recall@1': 0.0,
            'Recall@10': 0.0507, 'Recall@5': 0.0283}


def test_recommendation_evaluation_2(model, interactions_ds):
    """Evaluation with novelty=True."""
    assert recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=2, n_pos_interactions=None,
                                     novelty=True) == \
           {'HitRatio@2': 0.0233, 'NDCG@2': 0.0263, 'Precision@2': 0.03, 'Recall@2': 0.0233}


def test_recommendation_evaluation_3(model, interactions_ds):
    """Evaluation with limited number of positive interactions."""
    assert recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=2, n_pos_interactions=1,
                                     novelty=False) == \
           {'HitRatio@2': 0.02, 'NDCG@2': 0.0179, 'Precision@2': 0.01, 'Recall@2': 0.02}


def test_recommendation_evaluation_4(model, interactions_ds):
    """Evaluation with limited number of test users."""
    assert recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=2, n_pos_interactions=None,
                                     novelty=False, n_test_users=10) == \
           {'HitRatio@2': 0.0333, 'NDCG@2': 0.0585, 'Precision@2': 0.05, 'Recall@2': 0.0333}


def test_recommendation_evaluation_5(model):
    """Train evaluation."""
    assert recommendation_evaluation(model, cn_test_users=None, k=2, n_pos_interactions=None, novelty=False) == \
           {'HitRatio@2': 0.0176, 'NDCG@2': 0.0952, 'Precision@2': 0.23, 'Recall@2': 0.0176}


def test_recommendation_evaluation_6(model):
    """Train evaluation with novelty=True should result in all 0s."""
    assert recommendation_evaluation(model, cn_test_users=None, k=2, n_pos_interactions=None, novelty=True) == \
           {'HitRatio@2': 0.0, 'NDCG@2': 0.0, 'Precision@2': 0.0, 'Recall@2': 0.0}


def test_recommendation_evaluation_7(model, interactions_ds):
    """Evaluation with a custom interaction threshold."""
    assert recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=2, n_pos_interactions=None,
                                     novelty=False, interaction_threshold=2) == \
           {'HitRatio@2': 0.0069, 'NDCG@2': 0.0116, 'Precision@2': 0.0104, 'Recall@2': 0.0069}


def test_recommendation_evaluation_8(model, interactions_ds):
    """Evaluation with custom metrics."""
    assert recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=2, n_pos_interactions=None,
                                     novelty=False, metrics=[NDCG(), HitRatio()]) == \
           {'HitRatio@2': 0.0167, 'NDCG@2': 0.0189}


def test_recommendation_evaluation_9(model, interactions_ds):
    """Evaluation with custom metrics and k set to a list."""
    assert recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=[2, 3], n_pos_interactions=None,
                                     novelty=False, metrics=[NDCG(), HitRatio()], verbose=False) == \
           {'HitRatio@2': 0.0167, 'HitRatio@3': 0.0233, 'NDCG@2': 0.0189, 'NDCG@3': 0.022}


def test_recommendation_evaluation_10(model, interactions_ds):
    """Evaluation with invalid number of test users (0)."""
    try:
        recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=[2, 3], n_pos_interactions=None,
                                  novelty=False, n_test_users=0, metrics=[NDCG(), HitRatio()], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of test users (0) should be > 0.'


def test_recommendation_evaluation_11(model, interactions_ds):
    """Evaluation with invalid number of test users (< 0)."""
    try:
        recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=[1, 2], n_pos_interactions=None,
                                  novelty=False, n_test_users=-1,  metrics=[NDCG(), HitRatio()], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of test users (-1) should be > 0.'


def test_recommendation_evaluation_12(model, interactions_ds):
    """Evaluation with invalid number of positive interactions (0)."""
    try:
        recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=[1, 2], n_pos_interactions=0,
                                  novelty=False, metrics=[NDCG(), HitRatio()], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of positive interactions (0) should be None or an integer > 0.'


def test_recommendation_evaluation_13(model, interactions_ds):
    """Evaluation with invalid number of positive interactions (< 0)."""
    try:
        recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=[1, 2], n_pos_interactions=-1,
                                  novelty=False, metrics=[NDCG(), HitRatio()], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of positive interactions (-1) should be None or an integer > 0.'


def test_recommendation_evaluation_14(model, interactions_ds):
    """Evaluation with invalid number of k (0)."""
    try:
        recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=0, n_pos_interactions=None,
                                  novelty=False, metrics=[NDCG(), HitRatio()], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'k (0) should be > 0.'


def test_recommendation_evaluation_15(model, interactions_ds):
    """Evaluation with invalid number of k (< 0)."""
    try:
        recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=-1, n_pos_interactions=None,
                                  novelty=False, metrics=[NDCG(), HitRatio()], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'k (-1) should be > 0.'


def test_recommendation_evaluation_16(model, interactions_ds):
    """Invalid metrics value (not a list)."""
    try:
        recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=5, n_pos_interactions=None,
                                  novelty=False, metrics={}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'Expected "metrics" argument to be a list and found <class \'dict\'>. ' \
                         'Should contain instances of RankingMetricABC.'


def test_recommendation_evaluation_17(model, interactions_ds):
    """Invalid metrics value (list with non-RankingMetricABC instances)."""
    fun = lambda x: 1
    try:
        recommendation_evaluation(model, interactions_ds[1], n_test_users=None, k=5, n_pos_interactions=None,
                                  novelty=False, metrics=[fun], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == f'Expected metric {fun} to be an instance of type RankingMetricABC.'


def test_recommendation_evaluation_18(model, interactions_ds):
    """Evaluation with a custom ignore low predictions threshold."""
    assert recommendation_evaluation(model, interactions_ds[1], cn_test_users=None, k=2, n_pos_interactions=None,
                                     novelty=False, ignore_low_predictions_threshold=2) == \
            {'HitRatio@2': 0.0167, 'NDCG@2': 0.0189, 'Precision@2': 0.02, 'Recall@2': 0.0167}
