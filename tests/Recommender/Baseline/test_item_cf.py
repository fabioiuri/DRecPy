from DRecPy.Recommender.Baseline import ItemKNN
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
    fit_model = ItemKNN(k=20, m=5, sim_metric='adjusted_cosine', aggregation='weighted_mean', shrinkage=100,
                        use_averages=False)
    fit_model.fit(train_interaction_ds)
    return fit_model


@pytest.fixture(scope='module')
def fit_model_2(train_interaction_ds):
    fit_model_2 = ItemKNN(k=1, m=1, sim_metric='adjusted_cosine', aggregation='weighted_mean', shrinkage=100,
                          use_averages=False)
    fit_model_2.fit(train_interaction_ds)
    return fit_model_2


@pytest.fixture(scope='module')
def fit_model_use_averages(train_interaction_ds):
    fit_model_use_averages = ItemKNN(k=1, m=1, sim_metric='adjusted_cosine', aggregation='weighted_mean', shrinkage=100,
                                     use_averages=True)
    fit_model_use_averages.fit(train_interaction_ds)
    return fit_model_use_averages


@pytest.fixture(scope='module')
def fit_model_no_shrinkage(train_interaction_ds):
    fit_model_no_shrinkage = ItemKNN(k=20, m=5, sim_metric='adjusted_cosine', aggregation='weighted_mean',
                                     shrinkage=None, use_averages=False)
    fit_model_no_shrinkage.fit(train_interaction_ds)
    return fit_model_no_shrinkage


@pytest.fixture(scope='module')
def fit_model_cosine_sim(train_interaction_ds):
    fit_model_cosine_sim = ItemKNN(k=20, m=5, sim_metric='cosine', aggregation='weighted_mean', shrinkage=100,
                                   use_averages=False)
    fit_model_cosine_sim.fit(train_interaction_ds)
    return fit_model_cosine_sim


@pytest.fixture(scope='module')
def fit_model_mean_aggr(train_interaction_ds):
    fit_model_mean_aggr = ItemKNN(k=20, m=5, sim_metric='adjusted_cosine', aggregation='mean', shrinkage=100,
                                  use_averages=False)
    fit_model_mean_aggr.fit(train_interaction_ds)
    return fit_model_mean_aggr


def test_predict_0(fit_model):
    assert round(fit_model.predict(1, 2), 4) == 3.8031


def test_predict_1(fit_model_2):
    assert fit_model_2.predict(1, 2, skip_errors=True) is None


def test_predict_2(fit_model_use_averages):
    assert round(fit_model_use_averages.predict(1, 2), 4) == 3.6031


def test_predict_3(fit_model_no_shrinkage):
    assert round(fit_model_no_shrinkage.predict(1, 2), 4) == 3.1952


def test_predict_4(fit_model_cosine_sim):
    assert round(fit_model_cosine_sim.predict(1, 2), 4) == 4.2211


def test_predict_5(fit_model_mean_aggr):
    assert round(fit_model_mean_aggr.predict(1, 2), 4) == 3.9


def test_predict_6(fit_model, test_interaction_ds):
    predictions = [fit_model.predict(u, i, skip_errors=True)
                   for u, i in test_interaction_ds.values_list(['user', 'item'], to_list=True)[:100]]
    predictions = [p if p is not None else 0 for p in predictions]
    assert round(RMSE()(test_interaction_ds.values_list('interaction', to_list=True)[:100], predictions), 4) == 1.2019


def test_predict_7(fit_model_2, test_interaction_ds):
    predictions = [fit_model_2.predict(u, i, skip_errors=True)
                   for u, i in test_interaction_ds.values_list(['user', 'item'], to_list=True)[:100]]
    predictions = [p if p is not None else 0 for p in predictions]
    assert round(RMSE()(test_interaction_ds.values_list('interaction', to_list=True)[:100], predictions), 4) == 2.8142


def test_predict_8(fit_model_use_averages, test_interaction_ds):
    predictions = [fit_model_use_averages.predict(u, i, skip_errors=True)
                   for u, i in test_interaction_ds.values_list(['user', 'item'], to_list=True)[:100]]
    predictions = [p if p is not None else 0 for p in predictions]
    assert round(RMSE()(test_interaction_ds.values_list('interaction', to_list=True)[:100], predictions), 4) == 1.092


def test_predict_9(fit_model_no_shrinkage, test_interaction_ds):
    predictions = [fit_model_no_shrinkage.predict(u, i, skip_errors=True)
                   for u, i in test_interaction_ds.values_list(['user', 'item'], to_list=True)[:100]]
    predictions = [p if p is not None else 0 for p in predictions]
    assert round(RMSE()(test_interaction_ds.values_list('interaction', to_list=True)[:100], predictions), 4) == 1.7271


def test_predict_10(fit_model_cosine_sim, test_interaction_ds):
    predictions = [fit_model_cosine_sim.predict(u, i, skip_errors=True)
                   for u, i in test_interaction_ds.values_list(['user', 'item'], to_list=True)[:100]]
    predictions = [p if p is not None else 0 for p in predictions]
    assert round(RMSE()(test_interaction_ds.values_list('interaction', to_list=True)[:100], predictions), 4) == 1.2914


def test_predict_11(fit_model_mean_aggr, test_interaction_ds):
    predictions = [fit_model_mean_aggr.predict(u, i, skip_errors=True)
                   for u, i in test_interaction_ds.values_list(['user', 'item'], to_list=True)[:100]]
    predictions = [p if p is not None else 0 for p in predictions]
    assert round(RMSE()(test_interaction_ds.values_list('interaction', to_list=True)[:100], predictions), 4) == 1.1964


def test_recommend_0(fit_model):
    assert [(round(r, 4), item) for r, item in fit_model.recommend(1, 5)] == \
           [(5.0, 1296), (5.0, 1097), (5.0, 1405), (5.0, 1514), (5.0, 1489)]


def test_recommend_1(fit_model_2):
    assert [(round(r, 4), item) for r, item in fit_model_2.recommend(1, 5)] == \
           [(5.0, 1167), (5.0, 721), (5.0, 1294), (5.0, 1485), (5.0, 1439)]


def test_recommend_2(fit_model_use_averages):
    assert [(round(r, 4), item) for r, item in fit_model_use_averages.recommend(1, 5)] == \
           [(5.0, 1167), (5.0, 721), (5.0, 1294), (5.0, 1485), (5.0, 1439)]


def test_recommend_3(fit_model_no_shrinkage):
    assert [(round(r, 4), item) for r, item in fit_model_no_shrinkage.recommend(1, 5)] == \
           [(5.0, 1380), (5.0, 1234), (5.0, 1142), (5.0, 1097), (5.0, 61)]


def test_recommend_4(fit_model_cosine_sim):
    assert [(round(r, 4), item) for r, item in fit_model_cosine_sim.recommend(1, 5)] == \
           [(5.0, 1324), (5.0, 1176), (5.0, 889), (5.0, 749), (5.0, 1405)]


def test_recommend_5(fit_model_mean_aggr):
    assert [(round(r, 4), item) for r, item in fit_model_mean_aggr.recommend(1, 5)] == \
           [(5.0, 1405), (5.0, 1514), (5.0, 1489), (5.0, 1474), (5.0, 1442)]


def test_rank_0(fit_model, test_interaction_ds):
    ranked_list = [(round(r, 4), i) for r, i
                   in fit_model.rank(1, test_interaction_ds.values_list('item', to_list=True)[:5])]
    assert ranked_list == [(4.7573, 61), (4.5304, 20), (3.8902, 33), (3.8005, 155), (3.6394, 117)]


def test_rank_1(fit_model_2, test_interaction_ds):
    ranked_list = [(round(r, 4), i) for r, i
                   in fit_model_2.rank(1, test_interaction_ds.values_list('item', to_list=True)[:5])]
    assert ranked_list == [(5.0, 61), (3.0, 20)]


def test_rank_2(fit_model_use_averages, test_interaction_ds):
    ranked_list = [(round(r, 4), i) for r, i
                   in fit_model_use_averages.rank(1, test_interaction_ds.values_list('item', to_list=True)[:5])]
    assert ranked_list == [(5.0, 61), (3.6031, 155), (3.6031, 33), (3.6031, 117), (3.0, 20)]


def test_rank_3(fit_model_no_shrinkage, test_interaction_ds):
    ranked_list = [(round(r, 4), i) for r, i
                   in fit_model_no_shrinkage.rank(1, test_interaction_ds.values_list('item', to_list=True)[:5])]
    assert ranked_list == [(5.0, 61), (4.004, 33), (3.8901, 155), (3.7306, 20), (3.1486, 117)]


def test_rank_4(fit_model_cosine_sim, test_interaction_ds):
    ranked_list = [(round(r, 4), i) for r, i
                   in fit_model_cosine_sim.rank(1, test_interaction_ds.values_list('item', to_list=True)[:5])]
    assert ranked_list == [(4.4447, 61), (4.4415, 20), (4.3371, 33), (4.2026, 117), (3.9373, 155)]


def test_rank_5(fit_model_mean_aggr, test_interaction_ds):
    ranked_list = [(round(r, 4), i) for r, i
                   in fit_model_mean_aggr.rank(1, test_interaction_ds.values_list('item', to_list=True)[:5])]
    assert ranked_list == [(4.6667, 20), (4.625, 61), (3.8571, 33), (3.7273, 155), (3.6364, 117)]
