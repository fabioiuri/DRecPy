from DRecPy.Recommender.Baseline import UserKNN

from DRecPy.Dataset import get_train_dataset
from DRecPy.Dataset import get_test_dataset

from DRecPy.Evaluation.Metrics import rmse

df_train = get_train_dataset('ml-100k')
df_test = get_test_dataset('ml-100k')

fit_model = UserKNN()
fit_model.fit(df_train)
fit_model_best = UserKNN(k=30, m=2)
fit_model_best.fit(df_train)

""" __init__ """
def test_init_0():
    """Test default fitted property."""
    assert UserKNN().fitted is False


def test_init_1():
    """Test default mapping properties."""
    assert UserKNN().user_mapping is None
    assert UserKNN().user_mapping_inv is None
    assert UserKNN().item_mapping is None
    assert UserKNN().item_mapping_inv is None


def test_init_2():
    """Test if correct predicts_wo properties are set."""
    assert UserKNN().predicts_wo_item is False
    assert UserKNN().predicts_wo_user is True


def test_init_3():
    """Test if min_rating and max_rating are settable properties."""
    assert UserKNN().min_rating is None
    assert UserKNN().max_rating is None
    assert UserKNN(min_rating=2).min_rating == 2
    assert UserKNN(max_rating=3).max_rating == 3


def test_init_4():
    """Test if verbose is a settable property."""
    assert UserKNN().verbose is False
    assert UserKNN(verbose=True).verbose is True


def test_init_5():
    """Test if the similarities property is not set before model being fitted."""
    assert UserKNN().similarities is None


def test_init_6():
    """Test if the ratings property is not set before model being fitted."""
    assert UserKNN().ratings is None


def test_init_7():
    """Test if the user_items property is not set before model being fitted."""
    assert UserKNN().user_items is None


def test_init_8():
    """Test if the item_users property is not set before model being fitted."""
    assert UserKNN().item_users is None


def test_init_9():
    """Test if k and m properties are settable."""
    assert UserKNN(k=13).k == 13
    assert UserKNN(m=2).m == 2


def test_init_10():
    """Test supported similarity metrics and if error is thrown."""
    assert UserKNN(sim_metric='adjusted_cosine').sim_metric == 'adjusted_cosine'
    assert UserKNN(sim_metric='cosine').sim_metric == 'cosine'
    try:
        UserKNN(sim_metric='')
    except Exception as e:
        assert str(e) == 'There is no similarity metric corresponding to the name "".'


""" fit """
def test_fit_0():
    """Test if properties are set after fit."""
    assert fit_model.fitted is True
    # min, max ratings
    assert fit_model.min_rating == 1
    assert fit_model.max_rating == 5
    # raw - internal mappings
    assert fit_model.user_mapping is not None
    assert fit_model.user_mapping_inv is not None
    assert fit_model.item_mapping is not None
    assert fit_model.item_mapping_inv is not None


def test_fit_1():
    """Test if ratings, user_items and similarities properties are set after fit."""
    assert fit_model.ratings is not None
    assert fit_model.user_items is not None
    assert fit_model.item_users is not None
    assert fit_model.similarities is not None


""" predict """
def test_predict_0():
    """Test if error is thrown when trying to get predictions before fitting the model."""
    try:
        UserKNN().predict(0, 0)
    except Exception as e:
        assert str(e) == 'The model requires to be fitted before being able to make predictions.'


def test_predict_1():
    """Test if model is able to predict without a valid user."""
    assert round(fit_model.predict(None, '1'), 4) == 3.8597


def test_predict_2():
    """Test if error is thrown when an invalid item is given."""
    try:
        fit_model.predict(1, None)
    except Exception as e:
        assert str(e) == 'Item None was not found and the model requires doesn\'t support those predictions.'


def test_predict_3():
    """Test if no error is thrown when the skip_errors parameter is set to True."""
    assert fit_model.predict(1, None, skip_errors=True) == 3


def test_predict_4():
    """Test if prediction value is correct."""
    assert round(fit_model.predict(1, 2), 4) == 3.044


def test_predict_5():
    """Test rmse on predicting ratings on the ml-100k test set."""
    predictions = [fit_model.predict(u, i, skip_errors=True) for u, i, _, _ in df_test.values]
    assert round(rmse(df_test['rating'], predictions), 4) == 1.0235


def test_predict_6():
    """Test rmse on predicting ratings on the ml-100k test set, with modified model."""
    predictions = [fit_model_best.predict(u, i, skip_errors=True) for u, i, _, _ in df_test.values]
    assert round(rmse(df_test['rating'], predictions), 4) == 1.022


""" recommend """
def test_recommend_0():
    """Test if error is thrown when trying to get recommendations before fitting the model."""
    model = UserKNN()
    try:
        model.recommend(0, 5)
    except Exception as e:
        assert str(e) == 'The model requires to be fitted before being able to make predictions.'


def test_recommend_1():
    """Test if error is thrown when trying to get recommendations for invalid users."""
    try:
        fit_model.recommend(-1, 5)
    except Exception as e:
        assert str(e) == 'User -1 was not found.'


def test_recommend_2():
    """Test if recommendations are correct."""
    assert [(round(r, 4), item) for r, item in fit_model.recommend(1, 5)] == \
           [(5.0, '1656'), (5.0, '1599'), (5.0, '1536'), (5.0, '1500'), (5.0, '1467')]


def test_recommend_3():
    """Test if the threshold property filters out 'weak' recommendations."""
    assert [(round(r, 4), item) for r, item in fit_model.recommend(1, 5, threshold=5.1)] == []


def test_recommend_4():
    """Test if different recommendations are provided with a modified model."""
    assert [(round(r, 4), item) for r, item in fit_model_best.recommend(1, 5)] == \
           [(5.0, '1656'), (5.0, '1599'), (5.0, '1536'), (5.0, '1500'), (5.0, '1467')]


def test_recommend_5():
    """Test if more recommendations are given, by increasing the second parameter."""
    assert [(round(r, 4), item) for r, item in fit_model_best.recommend(15, 20)] == \
           [(5.0, '1656'), (5.0, '1599'), (5.0, '1536'), (5.0, '1500'), (5.0, '1467'),
            (5.0, '1293'), (5.0, '1201'), (5.0, '1189'), (5.0, '1122'), (5.0, '814'),
            (4.7143, '1449'), (4.6372, '64'), (4.6252, '408'), (4.5814, '169'), (4.5739, '98'),
            (4.557, '114'), (4.5434, '12'), (4.5392, '318'), (4.5, '1642'), (4.5, '1594')]


def test_recommend_6():
    """Test if non-novel recommendations are given, by setting the novelty parameter to False."""
    assert [(round(r, 4), item) for r, item in fit_model_best.recommend(15, 20, novelty=False)] == \
           [(5.0, '1656'), (5.0, '1599'), (5.0, '1536'), (5.0, '1500'), (5.0, '1467'),
            (5.0, '1293'), (5.0, '1201'), (5.0, '1189'), (5.0, '1122'), (5.0, '814'),
            (4.7143, '1449'), (4.6753, '50'), (4.6372, '64'), (4.6252, '408'), (4.5814, '169'),
            (4.5739, '98'), (4.557, '114'), (4.5434, '12'), (4.5392, '318'), (4.5, '1642')]


""" rank """
def test_rank_0():
    """Test if error is thrown when trying to get rankings before fitting the model."""
    model = UserKNN()
    try:
        model.rank(1, ['1065', '1104', '1149', '1242', '1487'])
    except Exception as e:
        assert str(e) == 'The model requires to be fitted before being able to make predictions.'


def test_rank_1():
    """Test if error is thrown when trying to get rankings for an invalid user."""
    try:
        fit_model.rank(-1, ['1065', '1104', '1149', '1242', '1487'])
    except Exception as e:
        assert str(e) == 'User -1 was not found.'


def test_rank_2():
    """Test if the resulting ranking is correct."""
    ranked_list = [(round(r, 4), i) for r, i in fit_model.rank(1, ['1065', '1104', '1149', '1242', '1487'])]
    assert ranked_list == [(3.8348, '1149'), (3.6955, '1065'), (2.75, '1242'), (2.3333, '1104'), (2.0, '1487')]


def test_rank_3():
    """Test if the n parameter is being taken into account."""
    ranked_list = [(round(r, 4), i) for r, i in fit_model.rank(1, ['1065', '1104', '1149', '1242', '1487'], n=2)]
    assert ranked_list == [(3.8348, '1149'), (3.6955, '1065')]


def test_rank_4():
    """Test if only novel ranked items are returned."""
    ranked_list = [(round(r, 4), i) for r, i in fit_model_best.rank(15, ['170', '754', '864', '199', '1149'])]
    assert ranked_list == [(4.4124, '199'), (4.3668, '170'), (3.8261, '1149')]


def test_rank_5():
    """Test if novel and non-novel ranked items are returned, when the novelty parameter is set to False."""
    ranked_list = [(round(r, 4), i) for r, i in fit_model_best.rank(15, ['170', '754', '864', '199', '1149'], novelty=False)]
    assert ranked_list == [(4.4124, '199'), (4.3668, '170'), (3.8261, '1149'), (3.3953, '754'), (3.2014, '864')]


def test_rank_6():
    """Test if invalid items are skipped and ranking for valid items is still returned."""
    ranked_list = [(round(r, 4), i) for r, i in fit_model.rank(15, ['170', '754', 'AA', '199', '-1'])]
    assert ranked_list == [(4.489, '199'), (4.3544, '170')]


def test_rank_7():
    """Test if error is thrown if the skip_invalid_items parameter is set to False."""
    try:
        fit_model.rank(15, ['170', '754', 'AA', '199', '-1'], skip_invalid_items=False)
    except Exception as e:
        assert str(e) == 'Item AA was not found.'
