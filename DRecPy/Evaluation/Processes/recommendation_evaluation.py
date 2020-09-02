from DRecPy.Evaluation.Metrics import RankingMetricABC
from DRecPy.Evaluation.Metrics import Precision
from DRecPy.Evaluation.Metrics import Recall
from DRecPy.Evaluation.Metrics import HitRatio
from DRecPy.Evaluation.Metrics import NDCG
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
from threading import Lock
import time

n_tasks_done = 0
n_tasks_lock = Lock()
metric_lock = Lock()


def recommendation_evaluation(model, ds_test=None, n_test_users=None, k=10, n_pos_interactions=None, novelty=False,
                              seed=0, max_concurrent_threads=4, **kwds):
    """Executes a recommendation evaluation process, where the given model will be evaluated under the provided settings.
    This function is not thread-safe (i.e. concurrent calls might produce unexpected results). Instead of trying this,
    increase the max_concurrent_threads argument to speed up the process (if you've the available cores).

    Args:
        model: An instance of a Recommender to be evaluated.
        ds_test: An optional test InteractionDataset. If none is provided, then the test data will be the model
            training data. Evaluating on train data is not ideal for assessing the model's performance.
        n_test_users: An optional integer representing the number of users to evaluate the produced rankings.
            Defaults to the number of unique users of the provided test dataset.
        k: An optional integer (or a list of integers) representing the truncation factor (keep the first k elements for
             each ranked list), which then affects the produced metric evaluation. Default: 10.
        n_pos_interactions: The number of positive interactions to sample into the list that contains the positive items
            to be considered when evaluating the model provided recommendations, for each user. If for a given user,
            there's less than n_pos_interactions positive interactions, the user's evaluation will be skipped.
            When this argument is not provided, all positive interactions on the test set from each user will be
            sampled. Default: None.
        interaction_threshold: The interaction value threshold to consider an interaction value positive or negative.
            All values above or equal interaction_threshold are considered positive, and all values bellow are
            considered negative. Default: model.interaction_threshold.
        novelty: A boolean indicating whether only novel recommendations should be taken into account or not.
            Default: False.
        metrics: An optional list containing instances of RankingMetricABC. Default: [Precision(), Recall(),
            HitRatio(), NDCG()].
        max_concurrent_threads: An optional integer representing the max concurrent threads to use. Default: 4.
        seed: An optional, integer representing the seed for the random number generator used to sample positive
            and negative interaction pairs. Default: 0.
        verbose: A boolean indicating whether state logs and a final graph should be produced or not. Default: true.
        block: A boolean indicating whether the displayed graph block code execution or not. Note that this graph is
            only displayed when verbose=True. Default: true.

    Returns:
        A dict containing each metric name mapping to the corresponding metric value.
    """
    assert n_test_users is None or n_test_users > 0, f'The number of test users ({n_test_users}) should be > 0.'
    assert n_pos_interactions is None or n_pos_interactions > 0,\
        f'The number of positive interactions ({n_pos_interactions}) should be None or an integer > 0.'

    interaction_threshold = kwds.get('interaction_threshold', model.interaction_threshold)

    if type(k) is not list: k = [k]
    for k_ in k: assert k_ > 0, f'k ({k_}) should be > 0.'

    if ds_test is None:
        ds_test = model.interaction_dataset

    metrics = kwds.get('metrics', [Precision(), Recall(), HitRatio(), NDCG()])

    assert isinstance(metrics, list), f'Expected "metrics" argument to be a list and found {type(metrics)}. ' \
        f'Should contain instances of RankingMetricABC.'

    for m in metrics:
        assert isinstance(m, RankingMetricABC), f'Expected metric {m} to be an instance of type RankingMetricABC.'

    metric_sums = {(m.name, k_): [0, 0] for m in metrics for k_ in k}  # list of (metric value sum, metric rankings count)
    num_users_made = 0

    unique_test_users_ds = ds_test.unique('user')
    n_test_users = len(unique_test_users_ds) if n_test_users is None else min(n_test_users, len(unique_test_users_ds))

    if kwds.get('verbose', True):
        _iter = tqdm(unique_test_users_ds.values(['user'], to_list=True),
                     total=n_test_users, desc='Starting user evaluation tasks', position=0, leave=True)
    else:
        _iter = unique_test_users_ds.values(['user'], to_list=True)

    global n_tasks_done
    n_tasks_done, n_tasks = 0, 0

    pool = ThreadPool(processes=max_concurrent_threads)
    for user in _iter:
        if num_users_made >= n_test_users: break  # reach max number of rankings

        pool.apply_async(_recommendation_evaluation_user, (model, user, ds_test, interaction_threshold,
                                                           n_pos_interactions, metrics, novelty, metric_sums, k,
                                                           random.Random(seed)))
        n_tasks += 1
        num_users_made += 1
        seed += 1

    pool.close()  # Done adding tasks

    if kwds.get('verbose', True):
        curr_done = 0
        pbar = tqdm(total=n_tasks, desc='Evaluating model recommendation performance', position=0, leave=True)
        while n_tasks_done <= n_tasks:
            pbar.update(n_tasks_done - curr_done)
            curr_done = n_tasks_done
            if n_tasks_done == n_tasks:
                break
            time.sleep(1)

    pool.join()  # Wait for all tasks to complete

    results = {m + f'@{k}': round(metric_sums[(m, k)][0] / metric_sums[(m, k)][1], 4)
        if metric_sums[(m, k)][1] > 0 else 0 for m, k in metric_sums}

    if kwds.get('verbose', True) and len(k) > 1:
        fig, axes = plt.subplots(1)
        fig.suptitle(f'Evaluation Metrics for {model.__class__.__name__}')

        axes.set_ylabel("Value", fontsize=12)
        axes.set_xlabel("k", fontsize=12)
        k = sorted(k)
        for m in metrics: axes.plot(k, [results[m.name + f'@{k_}'] for k_ in k], '--o', label=m.name)
        plt.legend()
        plt.show(block=kwds.get('block', True))

    return results


def _recommendation_evaluation_user(model, user, ds_test, interaction_threshold, n_pos_interactions, metrics,
                                    novelty, metric_sums, k, rng):
    """Gathers the user positive and negative interactions, applies a ranking on them, and evaluates the provided
    metrics, adding the results to the metric_sums structure."""
    global n_tasks_done
    user_ds = ds_test.select(f'user == {user}')

    # get positive interactions
    user_test_pos_ds = user_ds.select(f'interaction >= {interaction_threshold}')
    if n_pos_interactions is None:
        user_interacted_items = user_test_pos_ds.values_list(['item', 'interaction'])
    else:
        if len(user_test_pos_ds) < n_pos_interactions:  # not enough positive interactions
            with n_tasks_lock:
                n_tasks_done += 1
            return
        user_interacted_items = rng.sample(user_test_pos_ds.values_list(['item', 'interaction']), n_pos_interactions)

    best_item = None if len(user_interacted_items) == 0 else \
        max(user_interacted_items, key=lambda p: -p['interaction'])['item']
    user_interacted_items = [pair['item'] for pair in user_interacted_items]

    # join and shuffle all items
    if len(user_interacted_items) == 0:
        with n_tasks_lock:
            n_tasks_done += 1
        return

    recommendations = [item for _, item in model.recommend(user, n=max(k), novelty=novelty, skip_invalid_items=True,
                                                           interaction_threshold=model.min_interaction)]
    relevancies = {item: (user_ds.select_one(f'item == {item}', ['interaction'], to_list=True) or 0)
                   for item in set(user_interacted_items).union(set(recommendations))}

    # evaluate performance
    with metric_lock:
        for m in metrics:
            for k_ in k:
                param_names = m.__call__.__code__.co_varnames
                params = dict()
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
                    metric_sums[(m.name, k_)][0] += m(**params)
                    metric_sums[(m.name, k_)][1] += 1
                except: pass

    with n_tasks_lock:
        n_tasks_done += 1
