import random
from tqdm import tqdm
import math
import threading


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


def leave_k_out(interaction_dataset, k=1, min_user_interactions=0, seed=0, **kwds):
    """Dataset split method that uses a leave k out strategy. More specifically,
    for each user with more than k interactions, k interactions are randomly selected and taken out
    from the train set and put into the test set. This means that there are never users
    present in the test set that are not present in the train set.
    Also, users without at least min_user_interactions interactions, will not be sampled in either sets
    (i.e. they're removed).

    Args:
        interaction_dataset: A InteractionDataset instance containing the user-item interactions.
        k: Optional integer that represents the number of interactions, per user, to use in the
            test set (and to remove from the train set). Default: 1.
        min_user_interactions: Optional integer that represents the minimum number of interactions
            each user needs to have to be included in the train or test set. Default: 0.
        seed: An integer that is used as a seed value for the pseudorandom number generator.
            Default: 0.
        test_interaction_threshold: Optional float representing the minimum interaction value required
            to add a record to the test dataset. If this argument is missing, k records will
            be sampled without a minimum interaction value required.
        verbose: Optional boolean that indicates if a progress bar showing the splitting progress
            should be displayed or not. Default: True.

    Returns:
        Two InteractionDataset instances: the train and test interaction datasets in this order.
    """
    assert k > 0, f'The value of k ({k}) must be > 0.'

    test_interaction_threshold = kwds.get('test_interaction_threshold', None)

    interaction_dataset.assign_internal_ids()  # to speed up search

    train_rids_to_rem = []
    test_rids = []
    unique_uids_ds = interaction_dataset.unique('uid')
    threads = []

    if kwds.get('verbose', True):
        _iter = tqdm(unique_uids_ds.values(columns=['uid'], to_list=True),
                     total=len(unique_uids_ds), desc='Splitting dataset')
    else:
        _iter = unique_uids_ds.values(columns=['uid'], to_list=True)

    for uid in _iter:
        t = threading.Thread(target=_leave_k_out_user, args=(interaction_dataset, train_rids_to_rem, test_rids,
                                                             min_user_interactions, test_interaction_threshold, k, uid,
                                                             random.Random(seed)))
        seed += 1
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    interaction_dataset.remove_internal_ids()
    ds_test = interaction_dataset.drop(test_rids, keep=True)
    ds_train = interaction_dataset.drop(train_rids_to_rem + test_rids)

    return ds_train, ds_test


def _leave_k_out_user(interaction_dataset, train_rids_to_rem, test_rids, min_user_interactions,
                      test_interaction_threshold, k, uid, rng):
    user_rows_ds = interaction_dataset.select(f'uid == {uid}')

    if len(user_rows_ds) < min_user_interactions:  # not enough user interactions
        train_rids_to_rem.extend([record['rid'] for record in user_rows_ds.values(columns=['rid'])])
        return

    if len(user_rows_ds) <= k:  # not enough user interactions to sample
        return

    valid_records = user_rows_ds
    if test_interaction_threshold is not None:
        valid_records = user_rows_ds.select(f'interaction >= {test_interaction_threshold}')
        if len(valid_records) <= k:  # not enough user interactions to sample
            return

    sampled_idxs = rng.sample(range(len(valid_records)), k)
    for rid, i in zip(valid_records.values(columns=['rid'], to_list=True), range(len(valid_records))):
        if i in sampled_idxs:
            test_rids.append(rid)
