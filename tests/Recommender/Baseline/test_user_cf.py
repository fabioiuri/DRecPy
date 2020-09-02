from DRecPy.Recommender.Baseline import UserKNN
from DRecPy.Dataset import get_train_dataset
from DRecPy.Dataset import get_test_dataset
from DRecPy.Evaluation.Metrics import RMSE
import pytest


@pytest.fixture(scope='module')
def train_interaction_ds():
    return get_train_dataset('ml-100k')


@pytest.fixture(scope='module')
def test_interaction_ds():
    return get_test_dataset('ml-100k')


@pytest.fixture(scope='module')
def fit_model(train_interaction_ds):
    fit_model = UserKNN(k=20, m=5, sim_metric='adjusted_cosine', aggregation='weighted_mean', shrinkage=100,
                        use_averages=False)
    fit_model.fit(train_interaction_ds)
    return fit_model


@pytest.fixture(scope='module')
def fit_model_2(train_interaction_ds):
    fit_model_2 = UserKNN(k=1, m=1, sim_metric='adjusted_cosine', aggregation='weighted_mean', shrinkage=100,
                          use_averages=False)
    fit_model_2.fit(train_interaction_ds)
    return fit_model_2


@pytest.fixture(scope='module')
def fit_model_use_averages(train_interaction_ds):
    fit_model_use_averages = UserKNN(k=1, m=1, sim_metric='adjusted_cosine', aggregation='weighted_mean', shrinkage=100,
                                     use_averages=True)
    fit_model_use_averages.fit(train_interaction_ds)
    return fit_model_use_averages


@pytest.fixture(scope='module')
def fit_model_no_shrinkage(train_interaction_ds):
    fit_model_no_shrinkage = UserKNN(k=20, m=5, sim_metric='adjusted_cosine', aggregation='weighted_mean',
                                     shrinkage=None, use_averages=False)
    fit_model_no_shrinkage.fit(train_interaction_ds)
    return fit_model_no_shrinkage


@pytest.fixture(scope='module')
def fit_model_cosine_sim(train_interaction_ds):
    fit_model_cosine_sim = UserKNN(k=20, m=5, sim_metric='cosine', aggregation='weighted_mean', shrinkage=100,
                                   use_averages=False)
    fit_model_cosine_sim.fit(train_interaction_ds)
    return fit_model_cosine_sim


@pytest.fixture(scope='module')
def fit_model_mean_aggr(train_interaction_ds):
    fit_model_mean_aggr = UserKNN(k=20, m=5, sim_metric='adjusted_cosine', aggregation='mean', shrinkage=100,
                                  use_averages=False)
    fit_model_mean_aggr.fit(train_interaction_ds)
    return fit_model_mean_aggr


def test_predict_0(fit_model):
    assert round(fit_model.predict(1, 2), 4) == 3.065


def test_predict_1(fit_model_2):
    assert fit_model_2.predict(1, 2, skip_errors=True) is None


def test_predict_2(fit_model_use_averages):
    assert round(fit_model_use_averages.predict(1, 2), 4) == 3.1983


def test_predict_3(fit_model_no_shrinkage):
    assert round(fit_model_no_shrinkage.predict(1, 2), 4) == 3.1258


def test_predict_4(fit_model_cosine_sim):
    assert round(fit_model_cosine_sim.predict(1, 2), 4) == 3.3017


def test_predict_5(fit_model_mean_aggr):
    assert round(fit_model_mean_aggr.predict(1, 2), 4) == 3.0714


def test_predict_6(fit_model, test_interaction_ds):
    predictions = [fit_model.predict(u, i, skip_errors=True)
                   for u, i in test_interaction_ds.values_list(['user', 'item'], to_list=True)[:100]]
    predictions = [p if p is not None else 0 for p in predictions]
    assert round(RMSE()(test_interaction_ds.values_list('interaction', to_list=True)[:100], predictions), 4) == 1.1591
    assert round(RMSE()(test_interaction_ds.values_list('interaction', to_list=True)[:100], predictions), 4) == 1.1591


def test_predict_7(fit_model_2, test_interaction_ds):
    predictions = [fit_model_2.predict(u, i, skip_errors=True)
                   for u, i in test_interaction_ds.values_list(['user', 'item'], to_list=True)[:100]]
    predictions = [p if p is not None else 0 for p in predictions]
    assert round(RMSE()(test_interaction_ds.values_list('interaction', to_list=True)[:100], predictions), 4) == 2.7477


def test_predict_8(fit_model_use_averages, test_interaction_ds):
    predictions = [fit_model_use_averages.predict(u, i, skip_errors=True)
                   for u, i in test_interaction_ds.values_list(['user', 'item'], to_list=True)[:100]]
    predictions = [p if p is not None else 0 for p in predictions]
    assert round(RMSE()(test_interaction_ds.values_list('interaction', to_list=True)[:100], predictions), 4) == 1.1155


def test_predict_9(fit_model_no_shrinkage, test_interaction_ds):
    predictions = [fit_model_no_shrinkage.predict(u, i, skip_errors=True)
                   for u, i in test_interaction_ds.values_list(['user', 'item'], to_list=True)[:100]]
    predictions = [p if p is not None else 0 for p in predictions]
    assert round(RMSE()(test_interaction_ds.values_list('interaction', to_list=True)[:100], predictions), 4) == 1.2235


def test_predict_10(fit_model_cosine_sim, test_interaction_ds):
    predictions = [fit_model_cosine_sim.predict(u, i, skip_errors=True)
                   for u, i in test_interaction_ds.values_list(['user', 'item'], to_list=True)[:100]]
    predictions = [p if p is not None else 0 for p in predictions]
    assert round(RMSE()(test_interaction_ds.values_list('interaction', to_list=True)[:100], predictions), 4) == 1.0628


def test_predict_11(fit_model_mean_aggr, test_interaction_ds):
    predictions = [fit_model_mean_aggr.predict(u, i, skip_errors=True)
                   for u, i in test_interaction_ds.values_list(['user', 'item'], to_list=True)[:100]]
    predictions = [p if p is not None else 0 for p in predictions]
    assert round(RMSE()(test_interaction_ds.values_list('interaction', to_list=True)[:100], predictions), 4) == 1.1599


def test_recommend_0(fit_model):
    assert [(round(r, 4), item) for r, item in fit_model.recommend(1, 5)] == \
           [(5.0, 1514), (5.0, 1467), (5.0, 1168), (5.0, 1143), (5.0, 1114)]


def test_recommend_1(fit_model_2):
    assert [(round(r, 4), item) for r, item in fit_model_2.recommend(1, 5)] == \
           [(5.0, 1019), (5.0, 853), (5.0, 653), (5.0, 603), (5.0, 508)]


def test_recommend_2(fit_model_use_averages):
    assert [(round(r, 4), item) for r, item in fit_model_use_averages.recommend(1, 5)] == \
           [(5.0, 1019), (5.0, 853), (5.0, 653), (5.0, 603), (5.0, 508)]


def test_recommend_3(fit_model_no_shrinkage):
    assert [(round(r, 4), item) for r, item in fit_model_no_shrinkage.recommend(1, 5)] == \
           [(5.0, 963), (5.0, 1529), (5.0, 1514), (5.0, 1467), (5.0, 1367)]


def test_recommend_4(fit_model_cosine_sim):
    assert [(round(r, 4), item) for r, item in fit_model_cosine_sim.recommend(1, 5)] == \
           [(5.0, 1589), (5.0, 1169), (5.0, 1168), (5.0, 1143), (5.0, 1114)]


def test_recommend_5(fit_model_mean_aggr):
    assert [(round(r, 4), item) for r, item in fit_model_mean_aggr.recommend(1, 5)] == \
           [(5.0, 1514), (5.0, 1467), (5.0, 1168), (5.0, 1143), (5.0, 1114)]


def test_rank_0(fit_model, test_interaction_ds):
    ranked_list = [(round(r, 4), i) for r, i
                   in fit_model.rank(1, test_interaction_ds.values_list('item', to_list=True)[:5])]
    assert ranked_list == [(4.3271, 61), (3.5431, 117), (3.2585, 33), (2.6752, 20), (2.5501, 155)]


def test_rank_1(fit_model_2, test_interaction_ds):
    ranked_list = [(round(r, 4), i) for r, i
                   in fit_model_2.rank(1, test_interaction_ds.values_list('item', to_list=True)[:5])]
    assert ranked_list == [(4.0, 33), (4.0, 117)]


def test_rank_2(fit_model_use_averages, test_interaction_ds):
    ranked_list = [(round(r, 4), i) for r, i
                   in fit_model_use_averages.rank(1, test_interaction_ds.values_list('item', to_list=True)[:5])]
    assert ranked_list == [(4.0, 33), (4.0, 117)]


def test_rank_3(fit_model_no_shrinkage, test_interaction_ds):
    ranked_list = [(round(r, 4), i) for r, i
                   in fit_model_no_shrinkage.rank(1, test_interaction_ds.values_list('item', to_list=True)[:5])]
    assert ranked_list == [(4.419, 61), (3.6, 117), (3.2214, 33), (2.9947, 20), (2.3877, 155)]


def test_rank_4(fit_model_cosine_sim, test_interaction_ds):
    ranked_list = [(round(r, 4), i) for r, i
                   in fit_model_cosine_sim.rank(1, test_interaction_ds.values_list('item', to_list=True)[:5])]
    assert ranked_list == [(3.8092, 61), (3.5277, 117), (3.1474, 33), (3.0005, 20), (2.7127, 155)]


def test_rank_5(fit_model_mean_aggr, test_interaction_ds):
    ranked_list = [(round(r, 4), i) for r, i
                   in fit_model_mean_aggr.rank(1, test_interaction_ds.values_list('item', to_list=True)[:5])]
    assert ranked_list == [(4.3333, 61), (3.5, 117), (3.2308, 33), (2.6667, 20), (2.5455, 155)]
