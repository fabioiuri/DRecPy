import random
from tqdm import tqdm
import math
from threading import Thread
from heapq import heappush, heapreplace


def random_split(interaction_dataset, test_ratio=0.25, seed=None, **kwds):
    """Random split that creates a train set with (1-test_ratio)% of the total rows, and a test set with
    the other test_ratio% of the rows.
    No guarantees of users or items existing on both datasets are made,
    therefore cases like: user X exists on the test set but not on the train set might happen.
    The use of this split should be directed to models that support these types of behaviour.

    Args:
        interaction_dataset: A InteractionDataset instance containing the user-item interactions.
        test_ratio: A floating-point value representing the ratio of rows used for the test set.
            Default: 0.25.
        seed: An integer that is used as a seed value for the pseudorandom number generator.
            If none is given, no seed will be used.
        verbose: Optional boolean that indicates if a progress bar showing the splitting progress
            should be displayed or not. Default: True.

    Returns:
        Two InteractionDataset instances: the train and test interaction datasets in this order.
    """
    assert 0 < test_ratio < 1, 'The test_ratio argument must be in the ]0, 1[ range.'

    rng = random.Random(seed)

    test_rids = []

    total_rows = len(interaction_dataset)
    test_idxs = sorted(rng.sample(range(0, total_rows), math.floor(total_rows * test_ratio)))

    assert len(test_idxs) > 0, f'The test_ratio of {test_ratio} is not enough to split any row from the full dataset.'

    if kwds.get('verbose', True):
        _iter = tqdm(zip(range(len(interaction_dataset)), interaction_dataset.values('rid')),
                     total=len(interaction_dataset), desc='Splitting dataset')
    else:
        _iter = zip(range(len(interaction_dataset)), interaction_dataset.values('rid'))

    pointer = 0
    for idx, record in _iter:
        if idx == test_idxs[pointer]:
            pointer += 1
            test_rids.append(record['rid'])
            if pointer >= len(test_idxs): break

    ds_test = interaction_dataset.drop(test_rids, keep=True)
    ds_train = interaction_dataset.drop(test_rids)

    return ds_train, ds_test


def matrix_split(interaction_dataset, user_test_ratio=0.2, item_test_ratio=0.2, min_user_interactions=0,
                 seed=0, max_concurrent_threads=4, **kwds):
    """Dataset split method that uses a matrix split strategy. More specifically, item_test_ratio items from
    user_test_ratio users are sampled out of the full dataset and moved to the test set, while the missing items and
    users make the training set.

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
    assert 0 < user_test_ratio < 1, f'Invalid user_test_ratio of {user_test_ratio}: must be in the range (0, 1)'
    assert 0 < item_test_ratio < 1, f'Invalid item_test_ratio of {item_test_ratio}: must be in the range (0, 1)'

    rng = random.Random(seed)
    ds_unique_users = interaction_dataset.unique('user')
    test_users = rng.sample(ds_unique_users.values_list('user', to_list=True), math.floor(len(ds_unique_users) * user_test_ratio))
    ds_unique_items = interaction_dataset.unique('item')
    test_items = rng.sample(ds_unique_items.values_list('item', to_list=True), math.floor(len(ds_unique_items) * item_test_ratio))

    if kwds.get('verbose', True):
        _iter = tqdm(test_users, total=len(test_users), desc='Splitting dataset')
    else:
        _iter = test_users

    threads = []
    train_rids_to_rem, test_rids = [], []
    for user in _iter:
        t = Thread(target=_matrix_split_user, args=(interaction_dataset, train_rids_to_rem, test_rids, user,
                                                    min_user_interactions, test_items))
        threads.append(t)
        t.start()

        if len(threads) >= max_concurrent_threads: [t.join() for t in threads]

    if len(threads) > 0: [t.join() for t in threads]

    ds_test = interaction_dataset.drop(test_rids, keep=True)
    ds_train = interaction_dataset.drop(train_rids_to_rem + test_rids)

    return ds_train, ds_test


def _matrix_split_user(interaction_dataset, train_rids_to_rem, test_rids, user, min_user_interactions, test_items):
    user_rows_ds = interaction_dataset.select(f'user == {user}')

    if len(user_rows_ds) < min_user_interactions:  # not enough user interactions
        train_rids_to_rem.extend([rid for rid in user_rows_ds.values('rid', to_list=True)])
        return

    to_test = []
    for rid, item in user_rows_ds.values(['rid', 'item'], to_list=True):
        if item in test_items:
            to_test.append(rid)

    if len(to_test) == len(user_rows_ds): return  # no training records sampled

    test_rids.extend(to_test)


def leave_k_out(interaction_dataset, k=1, min_user_interactions=0, last_timestamps=False, seed=0,
                max_concurrent_threads=4, **kwds):
    """Dataset split method that uses a leave k out strategy. More specifically,
    for each user with more than k interactions, k interactions are randomly selected and taken out
    from the train set and put into the test set. This means that there are never users
    present in the test set that are not present in the train set.
    Also, users without at least min_user_interactions interactions, will not be sampled in either sets
    (i.e. they're removed).

    Args:
        interaction_dataset: A InteractionDataset instance containing the user-item interactions.
        k: Optional integer or float value: if k is an integer, then it represents the number of interactions, per user,
            to use in the test set (and to remove from the train set); if k is a float value (and between 0 and 1),
            it represents the percentage of interactions, per user, to use in the test set (and to remove from the
            train set). Default: 1.
        min_user_interactions: Optional integer that represents the minimum number of interactions
            each user needs to have to be included in the train or test set. Default: 0.
        max_concurrent_threads: An optional integer representing the max concurrent threads to use. Default: 4.
        seed: An integer that is used as a seed value for the pseudorandom number generator.
            Default: 0.
        verbose: Optional boolean that indicates if a progress bar showing the splitting progress
            should be displayed or not. Default: True.

    Returns:
        Two InteractionDataset instances: the train and test interaction datasets in this order.
    """
    assert k > 0, f'The value of k ({k}) must be > 0.'

    ratio_variant = isinstance(k, float)

    if ratio_variant and (k >= 1 or k <= 0):
        raise Exception('The k parameter should be in the (0, 1) range when it\'s used as the percentage of '
                        'interactions to sample to the test set, per user')

    unique_users_ds = interaction_dataset.unique('user')
    if kwds.get('verbose', True):
        _iter = tqdm(unique_users_ds.values('user', to_list=True), total=len(unique_users_ds), desc='Splitting dataset')
    else:
        _iter = unique_users_ds.values('user', to_list=True)

    threads = []
    train_rids_to_rem, test_rids = [], []
    for user in _iter:
        seed += 1
        if ratio_variant:
            t = Thread(target=_leave_k_out_user_ratio, args=(interaction_dataset, train_rids_to_rem, test_rids,
                                                             min_user_interactions, user, k, random.Random(seed),
                                                             last_timestamps))
        else:
            t = Thread(target=_leave_k_out_user_fixed, args=(interaction_dataset, train_rids_to_rem, test_rids,
                                                             min_user_interactions, user, k, random.Random(seed),
                                                             last_timestamps))
        threads.append(t)
        t.start()

        if len(threads) >= max_concurrent_threads: [t.join() for t in threads]

    if len(threads) > 0: [t.join() for t in threads]

    ds_test = interaction_dataset.drop(test_rids, keep=True)
    ds_train = interaction_dataset.drop(train_rids_to_rem + test_rids)

    return ds_train, ds_test


def _leave_k_out_user_ratio(interaction_dataset, train_rids_to_rem, test_rids, min_user_interactions, user, k, rng,
                            last_timestamps):
    user_rows_ds = interaction_dataset.select(f'user == {user}')

    if len(user_rows_ds) < min_user_interactions:  # not enough user interactions
        train_rids_to_rem.extend([rid for rid in user_rows_ds.values('rid', to_list=True)])
        return

    n_sampled_items = int(len(user_rows_ds) * k)
    if n_sampled_items == 0: return  # not enough user interactions to sample

    if last_timestamps:
        sampled_test_rids = []
        for (rid, timestamp), i in zip(user_rows_ds.values(['rid', 'timestamp'], to_list=True), range(len(user_rows_ds))):
            if len(sampled_test_rids) < n_sampled_items:
                heappush(sampled_test_rids, (timestamp, rid))
            else:
                heapreplace(sampled_test_rids, (timestamp, rid))
        test_rids.extend([rid for _, rid in sampled_test_rids])
    else:
        sampled_idxs = rng.sample(range(len(user_rows_ds)), n_sampled_items)
        for rid, i in zip(user_rows_ds.values('rid', to_list=True), range(len(user_rows_ds))):
            if i in sampled_idxs:
                test_rids.append(rid)


def _leave_k_out_user_fixed(interaction_dataset, train_rids_to_rem, test_rids, min_user_interactions, user, k, rng,
                            last_timestamps):
    user_rows_ds = interaction_dataset.select(f'user == {user}')

    if len(user_rows_ds) < min_user_interactions:  # not enough user interactions
        train_rids_to_rem.extend([rid for rid in user_rows_ds.values('rid', to_list=True)])
        return

    if len(user_rows_ds) <= k: return  # not enough user interactions to sample

    if last_timestamps:
        sampled_test_rids = []
        for (rid, timestamp), i in zip(user_rows_ds.values(['rid', 'timestamp'], to_list=True), range(len(user_rows_ds))):
            if len(sampled_test_rids) < k: heappush(sampled_test_rids, (timestamp, rid))
            else: heapreplace(sampled_test_rids, (timestamp, rid))
        test_rids.extend([rid for _, rid in sampled_test_rids])
    else:
        sampled_idxs = rng.sample(range(len(user_rows_ds)), k)
        for rid, i in zip(user_rows_ds.values('rid', to_list=True), range(len(user_rows_ds))):
            if i in sampled_idxs:
                test_rids.append(rid)
