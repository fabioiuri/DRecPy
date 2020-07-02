from DRecPy.Evaluation.Processes import ranking_evaluation
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


def test_ranking_evaluation_0(model, interactions_ds):
    """Evaluation without generating negative pairs."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=2, n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=False) == \
           {'AP@2': 0.53, 'HR@2': 0.3137, 'NDCG@2': 0.4093, 'P@2': 0.7021, 'R@2': 0.3137, 'RR@2': 0.28}


def test_ranking_evaluation_1(model, interactions_ds):
    """Evaluation with generated negative pairs."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=2, n_pos_interactions=None,
                              n_neg_interactions=20, generate_negative_pairs=True, novelty=False) == \
           {'AP@2': 0.125, 'HR@2': 0.0943, 'NDCG@2': 0.1249, 'P@2': 0.16, 'R@2': 0.0943, 'RR@2': 0.06}


def test_ranking_evaluation_2(model, interactions_ds):
    """Evaluation with limited negative pairs."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=2, n_pos_interactions=None,
                              n_neg_interactions=1, generate_negative_pairs=False, novelty=False) == \
           {'AP@2': 0.565, 'HR@2': 0.3337, 'NDCG@2': 0.4341, 'P@2': 0.8111, 'R@2': 0.3337, 'RR@2': 0.28}


def test_ranking_evaluation_3(model, interactions_ds):
    """Evaluation with k parameter set to a list."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=[1, 5, 10], n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=False, verbose=False) == \
           {'AP@1': 0.7, 'AP@10': 0.3668, 'AP@5': 0.3668, 'HR@1': 0.1953, 'HR@10': 0.4107, 'HR@5': 0.4107,
            'NDCG@1': 0.3968, 'NDCG@10': 0.4189, 'NDCG@5': 0.4189, 'P@1': 0.7447, 'P@10': 0.7089, 'P@5': 0.7089,
            'R@1': 0.1953, 'R@10': 0.4107, 'R@5': 0.4107, 'RR@1': 0.24, 'RR@10': 0.305, 'RR@5': 0.305}


def test_ranking_evaluation_4(model, interactions_ds):
    """Evaluation with k parameter set to a list and generated negative pairs."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=[1, 2, 3], n_pos_interactions=None,
                              n_neg_interactions=20, generate_negative_pairs=True, novelty=False, verbose=False) == \
           {'AP@1': 0.14, 'AP@2': 0.125, 'AP@3': 0.0978, 'HR@1': 0.0397, 'HR@2': 0.0943, 'HR@3': 0.1233,
            'NDCG@1': 0.0965, 'NDCG@2': 0.1249, 'NDCG@3': 0.1303, 'P@1': 0.14, 'P@2': 0.16, 'P@3': 0.14,
            'R@1': 0.0397, 'R@2': 0.0943, 'R@3': 0.1233, 'RR@1': 0.02, 'RR@2': 0.06, 'RR@3': 0.0667}


def test_ranking_evaluation_5(model, interactions_ds):
    """Evaluation with novelty=True."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=2, n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=True) == \
           {'AP@2': 0.53, 'HR@2': 0.3137, 'NDCG@2': 0.4093, 'P@2': 0.7021, 'R@2': 0.3137, 'RR@2': 0.28}


def test_ranking_evaluation_6(model, interactions_ds):
    """Evaluation with limited number of positive interactions."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=2, n_pos_interactions=1,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=False) == \
           {'AP@2': 0.43, 'HR@2': 0.46, 'NDCG@2': 0.3858, 'P@2': 0.4487, 'R@2': 0.46, 'RR@2': 0.43}


def test_ranking_evaluation_7(model, interactions_ds):
    """Evaluation with limited number of test users."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=10, k=2, n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=False) == \
           {'AP@2': 0.625, 'HR@2': 0.3383, 'NDCG@2': 0.4339, 'P@2': 0.75, 'R@2': 0.3383, 'RR@2': 0.4}


def test_ranking_evaluation_8(model):
    """Train evaluation."""
    assert ranking_evaluation(model, n_test_users=None, k=2, n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=False) == \
           {'AP@2': 0.835, 'HR@2': 0.0717, 'NDCG@2': 0.3845, 'P@2': 0.88, 'R@2': 0.0717, 'RR@2': 0.06}


def test_ranking_evaluation_9(model):
    """Train evaluation with novelty=True should result in all 0s."""
    assert ranking_evaluation(model, n_test_users=None, k=2, n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=True) == \
           {'AP@2': 0.0, 'HR@2': 0.0, 'NDCG@2': 0.0, 'P@2': 0, 'R@2': 0.0, 'RR@2': 0.0}


def test_ranking_evaluation_10(model, interactions_ds):
    """Evaluation with a custom interaction threshold."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=2, n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                              interaction_threshold=2) == \
           {'AP@2': 0.4219, 'HR@2': 0.3142, 'NDCG@2': 0.4093, 'P@2': 0.5638, 'R@2': 0.3142, 'RR@2': 0.16}


def test_ranking_evaluation_11(model, interactions_ds):
    """Evaluation with custom metrics."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=2, n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                              metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})}) == {'HR@2': 0.3137, 'NDCG@2': 0.4093}


def test_ranking_evaluation_12(model, interactions_ds):
    """Evaluation with custom metrics and k set to a list."""
    assert ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=[1, 2], n_pos_interactions=None,
                              n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                              metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})}, verbose=False) == \
           {'HR@1': 0.1953, 'HR@2': 0.3137, 'NDCG@1': 0.3968, 'NDCG@2': 0.4093}


def test_ranking_evaluation_13(model, interactions_ds):
    """Evaluation with invalid number of test users (0)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=0, k=[1, 2], n_pos_interactions=None,
                           n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                           metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of test users (0) should be > 0.'


def test_ranking_evaluation_14(model, interactions_ds):
    """Evaluation with invalid number of test users (< 0)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=-1, k=[1, 2], n_pos_interactions=None,
                           n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                           metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of test users (-1) should be > 0.'


def test_ranking_evaluation_15(model, interactions_ds):
    """Evaluation with invalid number of positive interactions (0)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=[1, 2], n_pos_interactions=0,
                           n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                           metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of positive interactions (0) should be None or an integer > 0.'


def test_ranking_evaluation_16(model, interactions_ds):
    """Evaluation with invalid number of positive interactions (< 0)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=[1, 2], n_pos_interactions=-1,
                           n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                           metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of positive interactions (-1) should be None or an integer > 0.'


def test_ranking_evaluation_17(model, interactions_ds):
    """Evaluation with invalid number of negative interactions (0)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=[1, 2], n_pos_interactions=None,
                           n_neg_interactions=0, generate_negative_pairs=False, novelty=False,
                           metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of negative interactions (0) should be None or an integer > 0.'


def test_ranking_evaluation_18(model, interactions_ds):
    """Evaluation with invalid number of negative interactions (< 0)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=[1, 2], n_pos_interactions=None,
                           n_neg_interactions=-1, generate_negative_pairs=False, novelty=False,
                           metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'The number of negative interactions (-1) should be None or an integer > 0.'


def test_ranking_evaluation_19(model, interactions_ds):
    """Evaluation with invalid combination of generate_negative_pairs and n_neg_interactions
    (generate_negative_pairs without a set value of n_neg_interactions)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=[1, 2], n_pos_interactions=None,
                           n_neg_interactions=None, generate_negative_pairs=True, novelty=False,
                           metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})}, verbose=False)
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
                           metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'k (0) should be > 0.'


def test_ranking_evaluation_21(model, interactions_ds):
    """Evaluation with invalid number of k (< 0)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=-1, n_pos_interactions=None,
                           n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                           metrics={'NDCG': (ndcg, {}), 'HR': (hit_ratio, {})}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'k (-1) should be > 0.'


def test_ranking_evaluation_22(model, interactions_ds):
    """Invalid metrics value (not a dict)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=5, n_pos_interactions=None,
                           n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                           metrics=[], verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'Expected "metrics" argument to be of type dict and found <class '"'list'"'>. ' \
                         'Should map metric names to a tuple containing the corresponding metric function and an ' \
                         'extra argument dict.'


def test_ranking_evaluation_23(model, interactions_ds):
    """Invalid metrics value (dict with non-tuple values)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=5, n_pos_interactions=None,
                           n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                           metrics={'A': ndcg}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'Expected metric A to map to a tuple containing the corresponding metric function and an ' \
                         'extra argument dict.'


def test_ranking_evaluation_24(model, interactions_ds):
    """Invalid metrics value (dict with tuple values containing non-callables on the first element)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=5, n_pos_interactions=None,
                           n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                           metrics={'A': (1, {})}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'Expected metric A to map to a tuple containing the corresponding metric function and an ' \
                         'extra argument dict.'


def test_ranking_evaluation_25(model, interactions_ds):
    """Invalid metrics value (dict with tuple values containing non-dicts on the second element)."""
    try:
        ranking_evaluation(model, interactions_ds[1], n_test_users=None, k=5, n_pos_interactions=None,
                           n_neg_interactions=None, generate_negative_pairs=False, novelty=False,
                           metrics={'A': (ndcg, [])}, verbose=False)
        assert False
    except Exception as e:
        assert str(e) == 'Expected metric A to map to a tuple containing the corresponding metric function and an ' \
                         'extra argument dict.'


def test_ranking_evaluation_26(model):
    """Train evaluation with very high number of negative interactions should result in all 0s due to skipped users."""
    assert ranking_evaluation(model, n_test_users=None, k=2, n_pos_interactions=None,
                              n_neg_interactions=100000, generate_negative_pairs=True, novelty=True) == \
           {'AP@2': 0.0, 'HR@2': 0.0, 'NDCG@2': 0.0, 'P@2': 0, 'R@2': 0.0, 'RR@2': 0.0}