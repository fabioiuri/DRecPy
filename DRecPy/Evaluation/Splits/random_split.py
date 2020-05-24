import random
from tqdm import tqdm
import math


def random_split(interaction_dataset, test_ratio=0.25, seed=0, **kwds):
    """Random split that creates a train set with (100-100*test_ratio)% of the total rows, and a test set with
    the other (100*test_ratio)% of the rows.
    No guarantees of users or items existing on both datasets are made,
    therefore cases like: user X exists on the test set but not on the train set might happen.
    The use of this split should be directed to models that support these types of behaviour.

    Args:
        interaction_dataset: A InteractionDataset instance containing the user-item interactions.
        test_ratio: A floating-point value representing the ratio of rows used for the test set.
            Default: 0.25.
        seed: An integer that is used as a seed value for the pseudorandom number generator.
            If none is given, no seed will be used. Default: 0.
        verbose: Optional boolean that indicates if a progress bar showing the splitting progress
            should be displayed or not. Default: True.

    Returns:
        Two InteractionDataset instances: the train and test interaction datasets in this order.
    """
    assert 0 < test_ratio < 1, 'The test_ratio argument must be in the (0, 1) range.'

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
