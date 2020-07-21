from multiprocessing.pool import ThreadPool
from threading import Lock
from heapq import heappush, heapreplace
import random
from tqdm import tqdm
import time

n_tasks_done = 0
n_tasks_lock = Lock()
train_rem_lock = Lock()
test_lock = Lock()


def leave_k_out(interaction_dataset, k=1, min_user_interactions=0, last_timestamps=False, timestamp_label='timestamp',
                seed=0, max_concurrent_threads=4, **kwds):
    """Dataset split method that uses a leave k out strategy. More specifically,
    for each user with more than k interactions, k interactions are randomly selected and taken out
    from the train set and put into the test set. This means that there are never users
    present in the test set that are not present in the train set.
    Also, users without at least min_user_interactions interactions, will not be sampled in either sets
    (i.e. they're removed).
    This function is not thread-safe (i.e. concurrent calls might produce unexpected results). Instead of trying this,
    increase the max_concurrent_threads argument to speed up the process (if you've the available cores).

    Args:
        interaction_dataset: A InteractionDataset instance containing the user-item interactions.
        k: Optional integer or float value: if k is an integer, then it represents the number of interactions, per user,
            to use in the test set (and to remove from the train set); if k is a float value (and between 0 and 1),
            it represents the percentage of interactions, per user, to use in the test set (and to remove from the
            train set). Default: 1.
        min_user_interactions: Optional integer that represents the minimum number of interactions
            each user needs to have to be included in the train or test set. Default: 0.
        last_timestamps: Optional boolean that indicates whether the test records should be sampled by last timestamps
            (using the column with the name passed in the timestamp_label argument). Default: False.
        timestamp_label: Optional string that corresponds to the name of the timestamp column on the
            interaction_dataset. This is only used when the last_timestamps argument is set to True.
            Default: 'timestamp'.
        max_concurrent_threads: An optional integer representing the max concurrent threads to use. Default: 4.
        seed: An integer that is used as a seed value for the pseudorandom number generator.
            Default: 0.
        verbose: Optional boolean that indicates if a progress bar showing the splitting progress
            should be displayed or not. Default: True.

    Returns:
        Two InteractionDataset instances: the train and test interaction datasets in this order.
    """
    assert k > 0, f'The value of k ({k}) must be > 0.'
    assert max_concurrent_threads > 0, f'The value of max_concurrent_threads ({max_concurrent_threads}) must be > 0.'

    ratio_variant = isinstance(k, float)

    if ratio_variant and k >= 1:
        raise Exception('The k parameter should be in the (0, 1) range when it\'s used as the percentage of '
                        'interactions to sample to the test set, per user. Current value: ' + str(k))

    unique_users_ds = interaction_dataset.unique('user')
    if kwds.get('verbose', True):
        _iter = tqdm(unique_users_ds.values('user', to_list=True), total=len(unique_users_ds), desc='Creating user split tasks')
    else:
        _iter = unique_users_ds.values('user', to_list=True)

    global n_tasks_done
    n_tasks_done, n_tasks = 0, 0

    pool = ThreadPool(processes=max_concurrent_threads)
    train_rids_to_rem, test_rids = [], []
    for user in _iter:
        seed += 1
        n_tasks += 1
        if ratio_variant:
            pool.apply_async(_leave_k_out_user_ratio, (interaction_dataset, train_rids_to_rem, test_rids,
                                                       min_user_interactions, user, k, random.Random(seed),
                                                       last_timestamps, timestamp_label))
        else:
            pool.apply_async(_leave_k_out_user_fixed, (interaction_dataset, train_rids_to_rem, test_rids,
                                                       min_user_interactions, user, k, random.Random(seed),
                                                       last_timestamps, timestamp_label))

    pool.close()  # Done adding tasks

    if kwds.get('verbose', True):
        curr_done = 0
        pbar = tqdm(total=n_tasks, desc='Splitting dataset')
        while n_tasks_done <= n_tasks:
            pbar.update(n_tasks_done - curr_done)
            curr_done = n_tasks_done
            if n_tasks_done == n_tasks:
                break
            time.sleep(1)

    pool.join()  # Wait for all tasks to complete

    ds_test = interaction_dataset.drop(test_rids, keep=True)
    ds_train = interaction_dataset.drop(train_rids_to_rem + test_rids)

    return ds_train, ds_test


def _leave_k_out_user_ratio(interaction_dataset, train_rids_to_rem, test_rids, min_user_interactions, user, k, rng,
                            last_timestamps, timestamp_label):
    user_rows_ds = interaction_dataset.select(f'user == {user}')
    n_sampled_items = int(len(user_rows_ds) * k)
    _leave_k_out_user(user_rows_ds, train_rids_to_rem, test_rids, min_user_interactions, n_sampled_items, rng,
                      last_timestamps, timestamp_label)


def _leave_k_out_user_fixed(interaction_dataset, train_rids_to_rem, test_rids, min_user_interactions, user, k, rng,
                            last_timestamps, timestamp_label):
    user_rows_ds = interaction_dataset.select(f'user == {user}')
    _leave_k_out_user(user_rows_ds, train_rids_to_rem, test_rids, min_user_interactions, k, rng,
                      last_timestamps, timestamp_label)


def _leave_k_out_user(user_rows_ds, train_rids_to_rem, test_rids, min_user_interactions, k, rng,
                            last_timestamps, timestamp_label):
    global n_tasks_done

    if len(user_rows_ds) < min_user_interactions:  # not enough user interactions
        with train_rem_lock:
            train_rids_to_rem.extend(user_rows_ds.values_list('rid', to_list=True))
    elif len(user_rows_ds) > k > 0:
        if last_timestamps:
            sampled_test_rids = []
            for i, (rid, timestamp) in enumerate(user_rows_ds.values_list(['rid', timestamp_label], to_list=True)):
                if len(sampled_test_rids) < k: heappush(sampled_test_rids, (timestamp, rid))
                else: heapreplace(sampled_test_rids, (timestamp, rid))
            with test_lock:
                test_rids.extend([rid for _, rid in sampled_test_rids])
        else:
            user_rows_rids = user_rows_ds.values_list('rid', to_list=True)
            with test_lock:
                test_rids.extend(rng.sample(user_rows_rids, k))

    with n_tasks_lock:
        n_tasks_done += 1
