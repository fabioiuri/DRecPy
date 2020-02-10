from DRecPy.Evaluation.Metrics import precision
from DRecPy.Evaluation.Metrics import recall
from DRecPy.Evaluation.Metrics import hit_ratio
from DRecPy.Evaluation.Metrics import ndcg
from DRecPy.Evaluation.Metrics import reciprocal_rank
from DRecPy.Evaluation.Metrics import average_precision
from DRecPy.Evaluation.Metrics import rmse
from DRecPy.Evaluation.Metrics import mse
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


def predictive_evaluation(model, ds_test, n_test_predictions=None, skip_errors=False, **kwds):
    if n_test_predictions is None: n_test_predictions = len(ds_test)

    assert n_test_predictions > 0, f'The number of test users ({n_test_predictions}) should be > 0.'

    metrics = kwds.get('metrics', {
        'RMSE': (rmse, {}),
        'MSE': (mse, {})
    })

    assert type(metrics) is dict, f'Expected "metrics" argument to be of type dict and found {type(metrics)}.' \
        f'Should map metric names to a tuple containing the corresponding metric function and an extra argument dict.'

    for m in metrics:
        err_msg = f'Expected metric {m} to map to a tuple containing the corresponding metric function and an extra argument dict.'
        assert type(metrics[m]) is tuple, err_msg
        assert callable(metrics[m][0]), err_msg
        assert type(metrics[m][1]) is dict, err_msg

    n_test_predictions = min(n_test_predictions, len(ds_test))
    if kwds.get('verbose', True):
        import time  # to avoid line issue
        time.sleep(0.1)

        _iter = tqdm(ds_test.values(columns=['user', 'item', 'interaction'], to_list=True),
                     total=n_test_predictions, desc='Evaluating model predictive performance', position=0, leave=True)
    else:
        _iter = ds_test.values(columns=['user', 'item', 'interaction'], to_list=True)

    num_predictions_made = 0
    y_pred = []
    y_true = []
    for user, item, interaction in _iter:
        if num_predictions_made >= n_test_predictions: break  # reach max number of predictions

        pred = model.predict(user, item, skip_errors=skip_errors)
        if pred is None: continue
        y_pred.append(pred)
        y_true.append(interaction)
        num_predictions_made += 1

    # evaluate performance
    metric_values = {}
    for m in metrics:
        params = {**metrics[m][1], 'y_true': y_true, 'y_pred': y_pred}
        metric_values[m] = metrics[m][0](**params)

    return metric_values


def ranking_evaluation(model, ds_test, n_test_users=None, pos_interactions=1, neg_interactions=99, k=10,
                       interaction_threshold=0, seed=None, **kwds):
    """

    Args:
        model:
        ds_test:
        n_test_users:
        pos_interactions:
        neg_interactions:
        k:
        interaction_threshold:
        seed:
        **kwds:

    Returns:

    """
    if n_test_users is None: n_test_users = 999999999999

    assert n_test_users > 0, f'The number of test users ({n_test_users}) should be > 0.'
    assert pos_interactions > 0, f'The number of positive interactions ({pos_interactions}) should be > 0.'
    assert neg_interactions > 0, f'The number of negative interactions ({neg_interactions}) should be > 0.'

    if type(k) is not list: k = [k]
    for k_ in k:
        assert k_ > 0, f'k ({k_}) should be > 0.'

    rng = random.Random(seed)

    metrics = kwds.get('metrics', {
        'P': (precision, {}),
        'R': (recall, {}),
        'HR': (hit_ratio, {}),
        'NDCG': (ndcg, {}),
        'RR': (reciprocal_rank, {}),
        'AP': (average_precision, {})
    })

    assert type(metrics) is dict, f'Expected "metrics" argument to be of type dict and found {type(metrics)}.' \
        f'Should map metric names to a tuple containing the corresponding metric function and an extra argument dict.'

    for m in metrics:
        err_msg = f'Expected metric {m} to map to a tuple containing the corresponding metric function and an extra argument dict.'
        assert type(metrics[m]) is tuple, err_msg
        assert callable(metrics[m][0]), err_msg
        assert type(metrics[m][1]) is dict, err_msg

    metric_sums = {(m, k_): 0 for m in metrics for k_ in k}
    num_rankings_made = 0

    unique_test_users_ds = ds_test.unique('user')
    n_test_users = min(n_test_users, len(unique_test_users_ds))
    if kwds.get('verbose', True):
        import time  # to avoid line issue
        time.sleep(0.1)
        
        _iter = tqdm(unique_test_users_ds.values(columns=['user'], to_list=True),
                     total=n_test_users, desc='Evaluating model ranking performance')
    else:
        _iter = unique_test_users_ds.values(columns=['user'], to_list=True)

    for user in _iter:
        if num_rankings_made >= n_test_users: break  # reach max number of rankings

        user_train_pos_ds = model.interaction_dataset.select(f'user == {user}, interaction > {interaction_threshold}')
        user_test_ds = ds_test.select(f'user == {user}')
        user_test_pos_ds = user_test_ds.select(f'interaction > {interaction_threshold}')

        if len(user_test_pos_ds) < pos_interactions: continue  # not enough positive interactions

        # get positive interactions
        user_interacted_items = list(user_test_pos_ds.values(columns=['item', 'interaction']))
        user_interacted_items = rng.sample(user_interacted_items, pos_interactions)

        relevancies = [pair['interaction'] for pair in user_interacted_items]
        best_item = sorted(user_interacted_items, key=lambda p: -p['interaction'])[0]['item']
        user_interacted_items = [pair['item'] for pair in user_interacted_items]

        # get negative interactions
        user_non_interacted_items = []
        while len(user_non_interacted_items) < neg_interactions:
            new_item = rng.randint(0, model.n_items - 1)
            if user_test_pos_ds.exists(f'item == {new_item}') or user_train_pos_ds.exists(f'item == {new_item}'):
                continue
            user_non_interacted_items.append(new_item)

        # join and shuffle all items
        all_items = user_interacted_items + user_non_interacted_items
        rng.shuffle(all_items)

        # rank according to model
        recommendations = [item for _, item in model.rank(user, all_items)]

        # evaluate performance
        for m in metrics:
            for k_ in k:
                param_names = metrics[m][0].__code__.co_varnames
                params = {**metrics[m][1]}
                for param_name in param_names:
                    if param_name == 'recommendations':
                        params[param_name] = recommendations
                    elif param_name == 'relevant_recommendations':
                        params[param_name] = user_interacted_items
                    elif param_name == 'relevant_recommendation':
                        params[param_name] = best_item
                    elif param_name == 'relevancies':
                        params[param_name] = relevancies
                    elif param_name == 'k':
                        params[param_name] = k_
                metric_sums[(m, k_)] += metrics[m][0](**params)

        num_rankings_made += 1

    results = {m + f'@{k}': metric_sums[(m, k)] / num_rankings_made for m, k in metric_sums}

    if kwds.get('verbose', True) and len(k) > 1:
        fig, axes = plt.subplots(1)
        fig.suptitle('Evaluation Metrics')

        axes.set_ylabel("Value", fontsize=12)
        axes.set_xlabel("k", fontsize=12)
        k = sorted(k)
        for m in metrics:
            axes.plot(k, [results[m + f'@{k_}'] for k_ in k], '--o', label=m)
        plt.legend()
        plt.show()

    return results
