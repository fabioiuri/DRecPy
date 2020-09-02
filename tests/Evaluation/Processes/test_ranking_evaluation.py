from DRecPy.Evaluation.Processes import ranking_evaluation
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


def test_ranking_evaluation_0(model, interactions_ds):
    """Evaluation without generating negative pairs."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=2, n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=False) == \
           {'HitRatio@2': 0.3137, 'NDCG@2': 0.4093, 'Precision@2': 0.7021, 'Recall@2': 0.3137}


def test_ranking_evaluation_1(model, interactions_ds):
    """Evaluation with generated negative pairs."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=2, n_pos_interactions=None,
                              n_neg_interactions=20, generate_negative_pairs=True, novelty=False) == \
           {'HitRatio@2': 0.0943, 'NDCG@2': 0.1249, 'Precision@2': 0.16, 'Recall@2': 0.0943}


def test_ranking_evaluation_2(model, interactions_ds):
    """Evaluation with limited negative pairs."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=2, n_pos_interactions=None,
                              n_neg_interactions=1, generate_negative_pairs=False, novelty=False) == \
           {'HitRatio@2': 0.3337, 'NDCG@2': 0.4341, 'Precision@2': 0.8111, 'Recall@2': 0.3337}


def test_ranking_evaluation_3(model, interactions_ds):
    """Evaluation with k parameter set to a list."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=[1, 5, 10], n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=False, verbose=False) == \
           {'HitRatio@1': 0.1953, 'HitRatio@10': 0.4107, 'HitRatio@5': 0.4107, 'NDCG@1': 0.3968, 'NDCG@10': 0.4189,
            'NDCG@5': 0.4189, 'Precision@1': 0.7447, 'Precision@10': 0.7089, 'Precision@5': 0.7089, 'Recall@1': 0.1953,
            'Recall@10': 0.4107, 'Recall@5': 0.4107}


def test_ranking_evaluation_4(model, interactions_ds):
    """Evaluation with k parameter set to a list and generated negative pairs."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=[1, 2, 3], n_pos_interactions=None,
                              n_neg_interactions=20, generate_negative_pairs=True, novelty=False, verbose=False) == \
           {'HitRatio@1': 0.0397, 'HitRatio@2': 0.0943, 'HitRatio@3': 0.1233, 'NDCG@1': 0.0965, 'NDCG@2': 0.1249,
            'NDCG@3': 0.1303, 'Precision@1': 0.14, 'Precision@2': 0.16, 'Precision@3': 0.14, 'Recall@1': 0.0397,
            'Recall@2': 0.0943, 'Recall@3': 0.1233}


def test_ranking_evaluation_5(model, interactions_ds):
    """Evaluation with novelty=True."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=2, n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=True) == \
           {'HitRatio@2': 0.3137, 'NDCG@2': 0.4093, 'Precision@2': 0.7021, 'Recall@2': 0.3137}


def test_ranking_evaluation_6(model, interactions_ds):
    """Evaluation with limited number of positive interactions."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=2, n_pos_interactions=1,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=False) == \
           {'HitRatio@2': 0.46, 'NDCG@2': 0.3858, 'Precision@2': 0.4487, 'Recall@2': 0.46}


def test_ranking_evaluation_7(model, interactions_ds):
    """Evaluation with limited number of test users."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=10, k=2, n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=False) == \
           {'HitRatio@2': 0.3383, 'NDCG@2': 0.4339, 'Precision@2': 0.75, 'Recall@2': 0.3383}


def test_ranking_evaluation_8(model):
    """Train evaluation."""
    assert ranking_evaluation(model, n_test_users=None, k=2, n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=False) == \
           {'HitRatio@2': 0.0717, 'NDCG@2': 0.3845, 'Precision@2': 0.88, 'Recall@2': 0.0717}


def test_ranking_evaluation_9(model):
    """Train evaluation with novelty=True should result in all 0s."""
    assert ranking_evaluation(model, n_test_users=None, k=2, n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=True) == \
           {'HitRatio@2': 0.0, 'NDCG@2': 0.0, 'Precision@2': 0, 'Recall@2': 0.0}


def test_ranking_evaluation_10(model, interactions_ds):
    """Evaluation with a custom interaction threshold."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=2, n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                              interaction_threshold=2) == \
           {'HitRatio@2': 0.3142, 'NDCG@2': 0.4093, 'Precision@2': 0.5638, 'Recall@2': 0.3142}


def test_ranking_evaluation_11(model, interactions_ds):
    """Evaluation with custom metrics."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=2, n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                              metrics=[HitRatio(), NDCG()]) == {'HitRatio@2': 0.3137, 'NDCG@2': 0.4093}


def test_ranking_evaluation_12(model, interactions_ds):
    """Evaluation with custom metrics and k set to a list."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=[1, 2], n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                              metrics=[HitRatio(), NDCG()], verbose=False) == \
           {'HitRatio@1': 0.1953, 'HitRatio@2': 0.3137, 'NDCG@1': 0.3968, 'NDCG@2': 0.4093}


def test_ranking_evaluation_13(model, interactions_ds):
    """Evaluation with invalid number of test users (0)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=0, k=[1, 2], n_pos_interactions=None,
                           n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                           metrics=[HitRatio(), NDCG()], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of test users (0) should be > 0.'


def test_ranking_evaluation_14(model, interactions_ds):
    """Evaluation with invalid number of test users (< 0)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=-1, k=[1, 2], n_pos_interactions=None,
                           n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                           metrics=[HitRatio(), NDCG()], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of test users (-1) should be > 0.'


def test_ranking_evaluation_15(model, interactions_ds):
    """Evaluation with invalid number of positive interactions (0)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=[1, 2], n_pos_interactions=0,
                           n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                           metrics=[HitRatio(), NDCG()], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of positive interactions (0) should be None or an integer > 0.'


def test_ranking_evaluation_16(model, interactions_ds):
    """Evaluation with invalid number of positive interactions (< 0)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=[1, 2], n_pos_interactions=-1,
                           n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                           metrics=[HitRatio(), NDCG()], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of positive interactions (-1) should be None or an integer > 0.'


def test_ranking_evaluation_17(model, interactions_ds):
    """Evaluation with invalid number of negative interactions (0)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=[1, 2], n_pos_interactions=None,
                           n_neg_interactions=0, generate_negative_pairs=False, novelty=False,
                           metrics=[HitRatio(), NDCG()], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of negative interactions (0) should be None or an integer > 0.'


def test_ranking_evaluation_18(model, interactions_ds):
    """Evaluation with invalid number of negative interactions (< 0)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=[1, 2], n_pos_interactions=None,
                           n_neg_interactions=-1, generate_negative_pairs=False, novelty=False,
                           metrics=[HitRatio(), NDCG()], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of negative interactions (-1) should be None or an integer > 0.'


def test_ranking_evaluation_19(model, interactions_ds):
    """Evaluation with invalid combination of generate_negative_pairs and n_neg_interactions
    (generate_negative_pairs without a set value of n_neg_interactions)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=[1, 2], n_pos_interactions=None,
                           n_neg_interactions=None, generate_negative_pairs=True, novelty=False,
                           metrics=[HitRatio(), NDCG()], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'Cannot generate negative interaction pairs when the number of negative interactions per ' \
                         'user is not defined. Either set generate_negative_pairs=False or define the ' \
                         'n_neg_interactions parameter.'


def test_ranking_evaluation_20(model, interactions_ds):
    """Evaluation with invalid number of k (0)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=0, n_pos_interactions=None,
                           n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                           metrics=[HitRatio(), NDCG()], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'k (0) should be > 0.'


def test_ranking_evaluation_21(model, interactions_ds):
    """Evaluation with invalid number of k (< 0)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=-1, n_pos_interactions=None,
                           n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                           metrics=[HitRatio(), NDCG()], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'k (-1) should be > 0.'


def test_ranking_evaluation_22(model, interactions_ds):
    """Invalid metrics value (not a list)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=5, n_pos_interactions=None,
                           n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                           metrics={}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'Expected "metrics" argument to be a list and found <class \'dict\'>. ' \
                         'Should contain instances of RankingMetricABC.'


def test_ranking_evaluation_23(model, interactions_ds):
    """Invalid metrics value (list with non-RankingMetricABC instances)."""
    fun = lambda x: 1
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=5, n_pos_interactions=None,
                           n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                           metrics=[fun], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == f'Expected metric {fun} to be an instance of type RankingMetricABC.'
