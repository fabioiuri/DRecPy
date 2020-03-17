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
from .utils import ThreadWithReturnValue


def predictive_evaluation(model, ds_test, n_test_predictions=None, skip_errors=False, **kwds):
    """Executes a predictive evaluation process, where the given model will be evaluated under the provided settings.

    Args:
        model: An instance of a Recommender to be evaluated.
        ds_test: An optional test InteractionDataset. If none is provided, then the test data will be the model
            training data. Evaluating on train data is not ideal for assessing the model's performance.
        n_test_predictions: An optional integer representing the number of predictions to evaluate.
            Default: predict for every (user, item) pair on the test dataset.
        skip_errors: A boolean indicating whether to ignore errors produced during the predict calls, or not.
            Default: False.
        metrics: An optional dict mapping the names of the metrics to a tuple containing the metric eval function as the
            first element, and the default arguments to call it as the second element.
            Eg: {'RMSE': (rmse, {beta: 1.2})}. Default: dict with the following metrics: root-mean-squared error and
            mean-squared error.
        verbose: A boolean indicating whether state logs should be produced or not. Default: true.

    Returns:
        A dict containing each metric name mapping to the corresponding metric value.
    """
    if n_test_predictions is None: n_test_predictions = len(ds_test)
    if ds_test is None: ds_test = model.interaction_dataset

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
        metric_fn = metrics[m][0]
        params = {**metrics[m][1], 'y_true': y_true, 'y_pred': y_pred}
        metric_values[m] = round(metric_fn(**params), 4)

    return metric_values


def ranking_evaluation(model, ds_test=None, n_test_users=None, n_pos_interactions=1, n_neg_interactions=99, k=10,
                       seed=0, **kwds):
    """Executes a ranking evaluation process, where the given model will be evaluated under the provided settings.

    Args:
        model: An instance of a Recommender to be evaluated.
        ds_test: An optional test InteractionDataset. If none is provided, then the test data will be the model
            training data. Evaluating on train data is not ideal for assessing the model's performance.
        n_test_users: An optional integer representing the number of users to evaluate the produced rankings.
            Defaults to the number of unique users of the provided test dataset.
        n_pos_interactions: The number of positive interactions to sample into the list that is going to be ranked and
            evaluated. Default: 1.
        n_neg_interactions:  The number of negative interactions to sample into the list that is going to be ranked and
            evaluated. Default: 99.
        interaction_threshold: The interaction value threshold to consider an interaction value positive or negative.
            All values above interaction_threshold are considered positive, and all values equal or bellow are
            considered negative. Default: model.interaction_threshold.
        k: An optional integer (or a list of integers) representing the truncation factor (keep the first k elements for
             each ranked list), which then affects the produced metric evaluation. Default: 10.
        seed: An optional, integer representing the seed for the random number generator used to sample positive
            and negative interaction pairs. Default: 0.
        metrics: An optional dict mapping the names of the metrics to a tuple containing the metric eval function as the
            first element, and the default arguments to call it as the second element.
            Eg: {'f_score': (f_score, {beta: 1.2})}. Default: dict with the following metrics: precision at k, recall
            at k, hit ratio at k, normalized discounted cumulative gain at k, reciprocal ranking at k, average
            precision at k.
        verbose: A boolean indicating whether state logs should be produced or not. Default: true.

    Returns:
        A dict containing each metric name mapping to the corresponding metric value.
    """
    assert n_test_users is None or n_test_users > 0, f'The number of test users ({n_test_users}) should be > 0.'
    assert n_pos_interactions > 0, f'The number of positive interactions ({n_pos_interactions}) should be > 0.'
    assert n_neg_interactions > 0, f'The number of negative interactions ({n_neg_interactions}) should be > 0.'

    interaction_threshold = kwds.get('interaction_threshold', model.interaction_threshold)

    if type(k) is not list: k = [k]
    for k_ in k:
        assert k_ > 0, f'k ({k_}) should be > 0.'

    train_evaluation = False
    if ds_test is None or ds_test == model.interaction_dataset:
        train_evaluation = True
        ds_test = model.interaction_dataset

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
    n_test_users = len(unique_test_users_ds) if n_test_users is None else min(n_test_users, len(unique_test_users_ds))

    if kwds.get('verbose', True):
        _iter = tqdm(unique_test_users_ds.values(columns=['user'], to_list=True),
                     total=n_test_users, desc='Evaluating model ranking performance', position=0, leave=True)
    else:
        _iter = unique_test_users_ds.values(columns=['user'], to_list=True)

    max_concurrent_threads = 10
    threads = []
    for user in _iter:
        if num_rankings_made >= n_test_users: break  # reach max number of rankings

        t = ThreadWithReturnValue(target=_ranking_evaluation_user,
                                  args=(model, user, ds_test, interaction_threshold, n_pos_interactions,
                                        n_neg_interactions, train_evaluation, metrics, metric_sums, k,
                                        random.Random(seed)))
        seed += 1
        threads.append(t)
        t.start()

        if len(threads) >= max_concurrent_threads:
            for t in threads:
                if t.join() is True:
                    num_rankings_made += 1
            threads = []

    if len(threads) > 0:
        for t in threads:
            if t.join() is True:
                num_rankings_made += 1

    results = {m + f'@{k}': round(metric_sums[(m, k)] / num_rankings_made, 4) for m, k in metric_sums}

    if kwds.get('verbose', True) and len(k) > 1:
        fig, axes = plt.subplots(1)
        fig.suptitle('Evaluation Metrics')

        axes.set_ylabel("Value", fontsize=12)
        axes.set_xlabel("k", fontsize=12)
        k = sorted(k)
        for m in metrics:
            axes.plot(k, [results[m + f'@{k_}'] for k_ in k], '--o', label=m)
        plt.show()

    return results


def _ranking_evaluation_user(model, user, ds_test, interaction_threshold, n_pos_interactions, n_neg_interactions,
                             train_evaluation, metrics, metric_sums, k, rng):
    """Gathers the user positive and negative interactions, applies a ranking on them, and evaluates the provided
    metrics, adding the results to the metric_sums structure."""
    # get positive interactions
    user_test_pos_ds = ds_test.select(f'user == {user}, interaction > {interaction_threshold}')
    if len(user_test_pos_ds) < n_pos_interactions: return False  # not enough positive interactions

    user_interacted_items = rng.sample(user_test_pos_ds.values_list(columns=['item', 'interaction']), n_pos_interactions)

    relevancies = [pair['interaction'] for pair in user_interacted_items]
    best_item = sorted(user_interacted_items, key=lambda p: -p['interaction'])[0]['item']
    user_interacted_items = [pair['item'] for pair in user_interacted_items]

    user_non_interacted_items = []
    if train_evaluation:
        user_train_pos_ds = user_test_pos_ds
    else:
        user_train_pos_ds = model.interaction_dataset.select(f'user == {user}, interaction > {interaction_threshold}')

    # get negative interactions
    while len(user_non_interacted_items) < n_neg_interactions:
        new_item = rng.randint(0, model.n_items - 1)
        if train_evaluation and user_train_pos_ds.exists(f'item == {new_item}'):
            continue
        elif not train_evaluation and \
                (user_test_pos_ds.exists(f'item == {new_item}') or user_train_pos_ds.exists(f'item == {new_item}')):
            continue
        user_non_interacted_items.append(new_item)

    # join and shuffle all items
    all_items = user_interacted_items + user_non_interacted_items
    rng.shuffle(all_items)

    # rank according to model
    recommendations = [item for _, item in model.rank(user, all_items, novelty=not train_evaluation)]
    if len(all_items) == 0:
        return False

    # evaluate performance
    for m in metrics:
        for k_ in k:
            metric_fn = metrics[m][0]
            param_names = metric_fn.__code__.co_varnames
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
            metric_sums[(m, k_)] += metric_fn(**params)

    return True
