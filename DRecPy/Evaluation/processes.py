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
from threading import Thread


def predictive_evaluation(model, ds_test, n_test_predictions=None, skip_errors=True, **kwds):
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
    y_pred, y_true = [], []
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
                       generate_negative_pairs=True, seed=0, max_concurrent_threads=4, **kwds):
    """Executes a ranking evaluation process, where the given model will be evaluated under the provided settings.

    Args:
        model: An instance of a Recommender to be evaluated.
        ds_test: An optional test InteractionDataset. If none is provided, then the test data will be the model
            training data. Evaluating on train data is not ideal for assessing the model's performance.
        n_test_users: An optional integer representing the number of users to evaluate the produced rankings.
            Defaults to the number of unique users of the provided test dataset.
        n_pos_interactions: The number of positive interactions to sample into the list that is going to be ranked and
            evaluated for each user. Default: 1.
        n_neg_interactions:  The number of negative interactions to sample into the list that is going to be ranked and
            evaluated for each user. Default: 99.
        interaction_threshold: The interaction value threshold to consider an interaction value positive or negative.
            All values above or equal interaction_threshold are considered positive, and all values bellow are
            considered negative. Default: model.interaction_threshold.
        k: An optional integer (or a list of integers) representing the truncation factor (keep the first k elements for
             each ranked list), which then affects the produced metric evaluation. Default: 10.
        generate_negative_pairs: An optional boolean that controls whether negative interaction pairs should also be
            generated (interaction pairs not present on the train or test data sets are also sampled) or not (i.e. only
            gathered from the test data set, where interaction values are bellow than the interaction_threshold).
            Default: True.
        metrics: An optional dict mapping the names of the metrics to a tuple containing the metric eval function as the
            first element, and the default arguments to call it as the second element.
            Eg: {'f_score': (f_score, {beta: 1.2})}. Default: dict with the following metrics: precision at k, recall
            at k, hit ratio at k, normalized discounted cumulative gain at k, reciprocal ranking at k, average
            precision at k.
        max_concurrent_threads: An optional integer representing the max concurrent threads to use. Default: 4.
        seed: An optional, integer representing the seed for the random number generator used to sample positive
            and negative interaction pairs. Default: 0.
        verbose: A boolean indicating whether state logs should be produced or not. Default: true.

    Returns:
        A dict containing each metric name mapping to the corresponding metric value.
    """
    assert n_test_users is None or n_test_users > 0, f'The number of test users ({n_test_users}) should be > 0.'
    assert n_pos_interactions is None or n_pos_interactions > 0,\
        f'The number of positive interactions ({n_pos_interactions}) should be None or an integer > 0.'
    assert n_pos_interactions is None or n_neg_interactions > 0, \
        f'The number of negative interactions ({n_neg_interactions}) should be None or an integer > 0.'

    if generate_negative_pairs and n_neg_interactions is None:
        raise Exception('Cannot generate negative interaction pairs when the number of negative interactions per user '
                        'is not defined. Either set generate_negative_pairs=False or define the n_neg_interactions '
                        'parameter.')

    interaction_threshold = kwds.get('interaction_threshold', model.interaction_threshold)

    if type(k) is not list: k = [k]
    for k_ in k: assert k_ > 0, f'k ({k_}) should be > 0.'

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

    metric_sums = {(m, k_): [0, 0] for m in metrics for k_ in k}  # list of (metric value sum, metric rankings count)
    num_users_made = 0

    unique_test_users_ds = ds_test.unique('user')
    n_test_users = len(unique_test_users_ds) if n_test_users is None else min(n_test_users, len(unique_test_users_ds))

    if kwds.get('verbose', True):
        _iter = tqdm(unique_test_users_ds.values(columns=['user'], to_list=True),
                     total=n_test_users, desc='Evaluating model ranking performance', position=0, leave=True)
    else:
        _iter = unique_test_users_ds.values(columns=['user'], to_list=True)

    threads = []
    for user in _iter:
        if num_users_made >= n_test_users: break  # reach max number of rankings

        t = Thread(target=_ranking_evaluation_user, args=(model, user, ds_test, interaction_threshold,
                                                          n_pos_interactions, n_neg_interactions, train_evaluation,
                                                          metrics, metric_sums, k, generate_negative_pairs,
                                                          random.Random(seed)))
        num_users_made += 1
        seed += 1
        threads.append(t)
        t.start()

        if len(threads) >= max_concurrent_threads: [t.join() for t in threads]

    if len(threads) > 0: [t.join() for t in threads]

    results = {m + f'@{k}': round(metric_sums[(m, k)][0] / metric_sums[(m, k)][1], 4) if metric_sums[(m, k)][1] > 0 else 0
               for m, k in metric_sums}

    if kwds.get('verbose', True) and len(k) > 1:
        fig, axes = plt.subplots(1)
        fig.suptitle(f'Evaluation Metrics for {model.__class__.__name__}')

        axes.set_ylabel("Value", fontsize=12)
        axes.set_xlabel("k", fontsize=12)
        k = sorted(k)
        for m in metrics: axes.plot(k, [results[m + f'@{k_}'] for k_ in k], '--o', label=m)
        plt.legend()
        plt.show()

    return results


def _ranking_evaluation_user(model, user, ds_test, interaction_threshold, n_pos_interactions, n_neg_interactions,
                             train_evaluation, metrics, metric_sums, k, generate_negative_pairs, rng):
    """Gathers the user positive and negative interactions, applies a ranking on them, and evaluates the provided
    metrics, adding the results to the metric_sums structure."""
    # get positive interactions
    user_test_pos_ds = ds_test.select(f'user == {user}, interaction >= {interaction_threshold}')
    if n_pos_interactions is None:
        user_interacted_items = user_test_pos_ds.values_list(columns=['item', 'interaction'])
    else:
        if len(user_test_pos_ds) < n_pos_interactions: return  # not enough positive interactions
        user_interacted_items = rng.sample(user_test_pos_ds.values_list(columns=['item', 'interaction']),
                                           n_pos_interactions)

    best_item = None if len(user_interacted_items) == 0 else max(user_interacted_items, key=lambda p: -p['interaction'])['item']
    relevancies = [pair['interaction'] for pair in user_interacted_items]
    user_interacted_items = [pair['item'] for pair in user_interacted_items]

    # get negative interactions
    negative_pairs_ds = ds_test.select(f'user == {user}, interaction < {interaction_threshold}')
    if n_neg_interactions is None:
        user_non_interacted_items = negative_pairs_ds.values_list(columns=['item'], to_list=True)
    else:
        user_non_interacted_items = rng.sample(negative_pairs_ds.values_list(columns=['item'], to_list=True),
                                               min(n_neg_interactions, len(negative_pairs_ds)))
        if len(user_non_interacted_items) < n_neg_interactions and generate_negative_pairs:
            if train_evaluation:
                user_train_pos_ds = user_test_pos_ds
            else:
                user_train_pos_ds = model.interaction_dataset \
                    .select(f'user == {user}, interaction >= {interaction_threshold}')

            while len(user_non_interacted_items) < n_neg_interactions:
                new_item = rng.randint(0, model.n_items - 1)
                if train_evaluation and user_train_pos_ds.exists(f'item == {new_item}'):
                    continue
                elif not train_evaluation and (user_test_pos_ds.exists(f'item == {new_item}') or
                                               user_train_pos_ds.exists(f'item == {new_item}')):
                    continue
                user_non_interacted_items.append(new_item)

    # join and shuffle all items
    all_items = user_interacted_items + user_non_interacted_items
    if len(all_items) == 0: return
    rng.shuffle(all_items)

    # rank according to model
    recommendations = [item for _, item in model.rank(user, all_items, novelty=False, skip_invalid_items=True)]

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
            try:
                metric_sums[(m, k_)][0] += metric_fn(**params)
                metric_sums[(m, k_)][1] += 1
            except: pass
