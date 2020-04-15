import random
from tqdm import tqdm
import math
from threading import Thread


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
        _iter = tqdm(zip(range(len(interaction_dataset)), interaction_dataset.values(columns=['rid'])),
                     total=len(interaction_dataset), desc='Splitting dataset')
    else:
        _iter = zip(range(len(interaction_dataset)), interaction_dataset.values(columns=['rid']))

    pointer = 0
    for idx, record in _iter:
        if idx == test_idxs[pointer]:
            pointer += 1
            if pointer >= len(test_idxs):
                break

    ds_test = interaction_dataset.drop(test_rids, keep=True)
    ds_train = interaction_dataset.drop(test_rids)

    return ds_train, ds_test


def leave_k_out(interaction_dataset, k=1, min_user_interactions=0, seed=0, max_concurrent_threads=4, **kwds):
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

    interaction_dataset.assign_internal_ids()  # to speed up search

    if ratio_variant:
        rng = random.Random(seed)
        ds_unique_uids = interaction_dataset.unique('uid')
        test_uids = rng.sample(ds_unique_uids.values_list('uid', to_list=True), math.floor(len(ds_unique_uids) * k))
        ds_unique_iids = interaction_dataset.unique('iid')
        test_iids = rng.sample(ds_unique_iids.values_list('iid', to_list=True), math.floor(len(ds_unique_iids) * k))

    unique_uids_ds = interaction_dataset.unique('uid')
    if kwds.get('verbose', True):
        _iter = tqdm(unique_uids_ds.values(
            columns=['uid'], to_list=True), total=len(unique_uids_ds), desc='Splitting dataset'
        )
    else:
        _iter = unique_uids_ds.values(columns=['uid'], to_list=True)

    threads = []
    train_rids_to_rem, test_rids = [], []
    for uid in _iter:
        if ratio_variant:
            if uid not in test_uids: continue

            t = Thread(target=_leave_k_out_user_ratio, args=(interaction_dataset, train_rids_to_rem, test_rids,
                                                             min_user_interactions, uid, test_iids))
        else:
            seed += 1
            t = Thread(target=_leave_k_out_user_fixed, args=(interaction_dataset, train_rids_to_rem, test_rids,
                                                             min_user_interactions, uid, k, random.Random(seed)))
        threads.append(t)
        t.start()

        if len(threads) >= max_concurrent_threads: [t.join() for t in threads]

    if len(threads) > 0: [t.join() for t in threads]

    interaction_dataset.remove_internal_ids()
    ds_test = interaction_dataset.drop(test_rids, keep=True)
    ds_train = interaction_dataset.drop(train_rids_to_rem + test_rids)

    return ds_train, ds_test


def _leave_k_out_user_ratio(interaction_dataset, train_rids_to_rem, test_rids, min_user_interactions, uid, test_iids):
    user_rows_ds = interaction_dataset.select(f'uid == {uid}')

    if len(user_rows_ds) < min_user_interactions:  # not enough user interactions
        train_rids_to_rem.extend([record['rid'] for record in user_rows_ds.values(columns=['rid'])])
        return

    to_test = []
    for rid, iid in user_rows_ds.values(columns=['rid', 'iid'], to_list=True):
        if iid in test_iids:
            to_test.append(rid)

    if len(to_test) == len(user_rows_ds): return  # no training records sampled

    test_rids.extend(to_test)


def _leave_k_out_user_fixed(interaction_dataset, train_rids_to_rem, test_rids, min_user_interactions, uid, k, rng):
    user_rows_ds = interaction_dataset.select(f'uid == {uid}')

    if len(user_rows_ds) < min_user_interactions:  # not enough user interactions
        train_rids_to_rem.extend([record['rid'] for record in user_rows_ds.values(columns=['rid'])])
        return

    if len(user_rows_ds) <= k: return  # not enough user interactions to sample

    sampled_idxs = rng.sample(range(len(user_rows_ds)), k)
    for rid, i in zip(user_rows_ds.values(columns=['rid'], to_list=True), range(len(user_rows_ds))):
        if i in sampled_idxs:
            test_rids.append(rid)
