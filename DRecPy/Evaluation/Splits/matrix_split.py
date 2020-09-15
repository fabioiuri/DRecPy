from multiprocessing.pool import ThreadPool
from threading import Lock
import random
from tqdm import tqdm
import math
import time

n_tasks_done = 0
n_tasks_lock = Lock()
train_rem_lock = Lock()
test_lock = Lock()


def matrix_split(interaction_dataset, user_test_ratio=0.2, item_test_ratio=0.2, min_user_interactions=0,
                 seed=0, max_concurrent_threads=4, **kwds):
    """Dataset split method that uses a matrix split strategy. More specifically, item_test_ratio items from
    user_test_ratio users are sampled out of the full dataset and moved to the test set, while the missing items and
    users make the training set. If all records for a given user are selected to be moved into the test set, the split
    for that user is skipped, and its records are kept in the train set.
    This function is not thread-safe (i.e. concurrent calls might produce unexpected results). Instead of trying this,
    increase the max_concurrent_threads argument to speed up the process (if you've the available cores).

    Args:
        interaction_dataset: A InteractionDataset instance containing the user-item interactions.
        user_test_ratio: Optional float value that represents the percentage of users to be sampled to the test set.
        item_test_ratio: Optional float value that represents the percentage of items to be sampled to the test set.
        min_user_interactions: Optional integer that represents the minimum number of interactions each user needs
            to have to be included in the train or test set. Default: 0.
        max_concurrent_threads: An optional integer representing the max concurrent threads to use. Default: 4.
        seed: An integer that is used as a seed value for the pseudorandom number generator.
            Default: 0.
        verbose: Optional boolean that indicates if a progress bar showing the splitting progress
            should be displayed or not. Default: True.

    Returns:
        Two InteractionDataset instances: the train and test interaction datasets in this order.
    """
    assert 0 < user_test_ratio <= 1, f'Invalid user_test_ratio of {user_test_ratio}: must be in the range (0, 1]'
    assert 0 < item_test_ratio <= 1, f'Invalid item_test_ratio of {item_test_ratio}: must be in the range (0, 1]'
    assert max_concurrent_threads > 0, f'The value of max_concurrent_threads ({max_concurrent_threads}) must be > 0.'

    rng = random.Random(seed)
    ds_unique_users = interaction_dataset.unique('user')
    all_users = ds_unique_users.values_list('user', to_list=True)
    test_users_set = set(rng.sample(all_users, math.floor(len(all_users) * user_test_ratio)))

    ds_unique_items = interaction_dataset.unique('item')
    test_items_set = set(rng.sample(ds_unique_items.values_list('item', to_list=True),
                                    math.floor(len(ds_unique_items) * item_test_ratio)))

    if kwds.get('verbose', True):
        _iter = tqdm(all_users, total=len(all_users), desc='Creating user split tasks')
    else:
        _iter = all_users

    global n_tasks_done
    n_tasks_done, n_tasks = 0, 0

    pool = ThreadPool(processes=max_concurrent_threads)
    train_rids_to_rem, test_rids = [], []
    for user in _iter:
        n_tasks += 1
        pool.apply_async(_matrix_split_user, (interaction_dataset, train_rids_to_rem, test_rids, user,
                                              min_user_interactions, test_items_set, test_users_set))

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


def _matrix_split_user(interaction_dataset, train_rids_to_rem, test_rids, user, min_user_interactions, test_items_set,
                       test_users_set):
    global n_tasks_done
    user_rows_ds = interaction_dataset.select(f'user == {user}')

    if len(user_rows_ds) < min_user_interactions:  # not enough user interactions
        with train_rem_lock:
            train_rids_to_rem.extend([rid for rid in user_rows_ds.values('rid', to_list=True)])
    elif user in test_users_set:
        to_test = []
        for rid, item in user_rows_ds.values(['rid', 'item'], to_list=True):
            with test_lock:
                if item in test_items_set:
                    to_test.append(rid)

        if len(to_test) < len(user_rows_ds):
            with test_lock:
                test_rids.extend(to_test)

    with n_tasks_lock:
        n_tasks_done += 1
