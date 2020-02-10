from DRecPy.Recommender.Baseline import ItemKNN

from DRecPy.Dataset import get_train_dataset
from DRecPy.Dataset import get_test_dataset

from DRecPy.Evaluation.Metrics import rmse

df_train = get_train_dataset('ml-100k')
df_test = get_test_dataset('ml-100k')

fit_model = ItemKNN()
fit_model.fit(df_train)
fit_model_best = ItemKNN(k=10, m=3)
fit_model_best.fit(df_train)

""" __init__ """
def test_init_0():
    """Test default fitted property."""
    assert ItemKNN().fitted is False


def test_init_1():
    """Test default mapping properties."""
    assert ItemKNN().user_mapping is None
    assert ItemKNN().user_mapping_inv is None
    assert ItemKNN().item_mapping is None
    assert ItemKNN().item_mapping_inv is None


def test_init_2():
    """Test if correct predicts_wo properties are set."""
    assert ItemKNN().predicts_wo_item is True
    assert ItemKNN().predicts_wo_user is False


def test_init_3():
    """Test if min_rating and max_rating are settable properties."""
    assert ItemKNN().min_rating is None
    assert ItemKNN().max_rating is None
    assert ItemKNN(min_rating=2).min_rating == 2
    assert ItemKNN(max_rating=3).max_rating == 3


def test_init_4():
    """Test if verbose is a settable property."""
    assert ItemKNN().verbose is False
    assert ItemKNN(verbose=True).verbose is True


def test_init_5():
    """Test if the similarities property is not set before model being fitted."""
    assert ItemKNN().similarities is None


def test_init_6():
    """Test if the ratings property is not set before model being fitted."""
    assert ItemKNN().ratings is None


def test_init_7():
    """Test if the user_items property is not set before model being fitted."""
    assert ItemKNN().user_items is None


def test_init_8():
    """Test if k and m properties are settable."""
    assert ItemKNN(k=13).k == 13
    assert ItemKNN(m=2).m == 2


def test_init_9():
    """Test supported similarity metrics and if error is thrown."""
    assert ItemKNN(sim_metric='adjusted_cosine').sim_metric == 'adjusted_cosine'
    assert ItemKNN(sim_metric='cosine').sim_metric == 'cosine'
    try:
        ItemKNN(sim_metric='')
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
    assert fit_model.similarities is not None


""" predict """
def test_predict_0():
    """Test if error is thrown when trying to get predictions before fitting the model."""
    try:
        ItemKNN().predict(0, 0)
    except Exception as e:
        assert str(e) == 'The model requires to be fitted before being able to make predictions.'


def test_predict_1():
    """Test if model is able to predict without a valid item."""
    assert round(fit_model.predict(1, None), 4) == 3.6031


def test_predict_2():
    """Test if error is thrown when an invalid user is given."""
    try:
        fit_model.predict(None, 1)
    except Exception as e:
        assert str(e) == 'User None was not found and the model requires doesn\'t support those predictions.'


def test_predict_3():
    """Test if no error is thrown when the skip_errors parameter is set to True."""
    assert fit_model.predict(None, 1, skip_errors=True) == 3


def test_predict_4():
    """Test if prediction value is correct."""
    assert round(fit_model.predict(1, 2), 4) == 3.3105


def test_predict_5():
    """Test rmse on predicting ratings on the ml-100k test set."""
    predictions = [fit_model.predict(u, i) for u, i, _, _ in df_test.values]
    assert round(rmse(df_test['rating'], predictions), 4) == 0.9941


def test_predict_6():
    """Test rmse on predicting ratings on the ml-100k test set, with modified model."""
    predictions = [fit_model_best.predict(u, i) for u, i, _, _ in df_test.values]
    assert round(rmse(df_test['rating'], predictions), 4) == 0.9838


""" recommend """
def test_recommend_0():
    """Test if error is thrown when trying to get recommendations before fitting the model."""
    model = ItemKNN()
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
           [(4.7528, '1108'), (4.7297, '1252'), (4.6821, '1174'), (4.6803, '766'), (4.641, '1202')]


def test_recommend_3():
    """Test if the threshold property filters out 'weak' recommendations."""
    assert [(round(r, 4), item) for r, item in fit_model.recommend(1, 5, threshold=4.7)] == \
           [(4.7528, '1108'), (4.7297, '1252')]


def test_recommend_4():
    """Test if different recommendations are provided with a modified model."""
    assert [(round(r, 4), item) for r, item in fit_model_best.recommend(1, 5)] == \
           [(4.9358, '1242'), (4.8143, '1104'), (4.8069, '1065'), (4.8021, '1149'), (4.7934, '1487')]


def test_recommend_5():
    """Test if more recommendations are given, by increasing the second parameter."""
    assert [(round(r, 4), item) for r, item in fit_model_best.recommend(15, 10)] == \
           [(4.2295, '170'), (4.159, '863'), (4.0784, '199'), (4.0738, '961'), (4.0624, '880'),
            (4.0585, '794'), (4.04, '511'), (4.0049, '663'), (4.0005, '1121'), (3.9896, '1147')]


def test_recommend_6():
    """Test if non-novel recommendations are given, by setting the novelty parameter to False."""
    assert [(round(r, 4), item) for r, item in fit_model_best.recommend(15, 10, novelty=False)] == \
           [(4.2295, '170'), (4.2009, '754'), (4.165, '864'), (4.159, '863'), (4.1386, '459'),
            (4.1271, '676'), (4.1096, '936'), (4.0964, '620'), (4.0784, '199'), (4.0738, '961')]


""" rank """
def test_rank_0():
    """Test if error is thrown when trying to get rankings before fitting the model."""
    model = ItemKNN()
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
    assert ranked_list == [(4.5368, '1065'), (4.5203, '1149'), (3.6031, '1487'), (3.6031, '1242'), (3.6031, '1104')]


def test_rank_3():
    """Test if the n parameter is being taken into account."""
    ranked_list = [(round(r, 4), i) for r, i in fit_model.rank(1, ['1065', '1104', '1149', '1242', '1487'], n=2)]
    assert ranked_list == [(4.5368, '1065'), (4.5203, '1149')]


def test_rank_4():
    """Test if only novel ranked items are returned."""
    ranked_list = [(round(r, 4), i) for r, i in fit_model_best.rank(15, ['170', '754', '864', '199', '1149'])]
    assert ranked_list == [(4.2295, '170'), (4.0784, '199'), (3.2036, '1149')]


def test_rank_5():
    """Test if novel and non-novel ranked items are returned, when the novelty parameter is set to False."""
    ranked_list = [(round(r, 4), i) for r, i in fit_model_best.rank(15, ['170', '754', '864', '199', '1149'], novelty=False)]
    assert ranked_list == [(4.2295, '170'), (4.2009, '754'), (4.165, '864'), (4.0784, '199'), (3.2036, '1149')]


def test_rank_6():
    """Test if invalid items are skipped and ranking for valid items is still returned."""
    ranked_list = [(round(r, 4), i) for r, i in fit_model.rank(15, ['170', '754', 'AA', '199', '-1'])]
    assert ranked_list == [(3.9055, '199'), (3.8365, '170')]


def test_rank_7():
    """Test if error is thrown if the skip_invalid_items parameter is set to False."""
    try:
        fit_model.rank(15, ['170', '754', 'AA', '199', '-1'], skip_invalid_items=False)
    except Exception as e:
        assert str(e) == 'Item AA was not found.'
