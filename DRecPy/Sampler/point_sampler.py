from DRecPy.Dataset import InteractionDatasetABC
import random


class PointSampler:
    """PointSampler class that abstracts sampling positive and negative (user, item) interaction pairs.

    Args:
        interaction_dataset: An instance of the InteractionDataset type, representing the data set where points
            are being sampled from.
        neg_ratio: The number of negative interaction pairs to be sampled for each positive interaction pair.
        interaction_threshold: An optional integer that is used as the boundary interaction value between positive and
            negative interaction pairs. All values above or equal to interaction_threshold are considered positive,
            and all values bellow are considered negative. If none is provided, positive interactions are the ones
            present on the data set, and all the others are considered negative. Default: None.
        seed: An optional integer to be used as the seed value for the pseudo-random number generated used to sample
            interaction pairs. Default: None.
    """
    def __init__(self, interaction_dataset, neg_ratio, interaction_threshold=None, seed=None):
        assert interaction_dataset is not None, 'An interaction dataset instance is required.'
        assert neg_ratio is not None, 'A neg_ratio value is required.'

        assert InteractionDatasetABC in interaction_dataset.__class__.__bases__, f'Provided interaction_dataset  ' \
            f'argument is not subclass of InteractionDataset (found type {type(interaction_dataset)}).'

        assert interaction_dataset.has_internal_ids, \
            'The provided interaction dataset instance does not have internal ids assigned.'

        self.interaction_dataset = interaction_dataset
        self.neg_ratio = neg_ratio
        self.interaction_threshold = interaction_threshold
        self.rng = random.Random(seed)

        self.null_pair_gen = self.interaction_dataset.null_interaction_pair_generator(
            seed=seed, interaction_threshold=interaction_threshold
        )
        if self.interaction_threshold is None:
            self.pos_pair_gen = self.interaction_dataset.select_random_generator(seed=seed)
        else:
            self.pos_pair_gen = self.interaction_dataset.select_random_generator(
                f'interaction >= {interaction_threshold}', seed=seed
            )

    def sample(self, n=16):
        """Sample positive and negative interaction pairs according to the set definitions of the PointSampler instance.

        Args:
            n: An integer representing the number of interaction pairs to be sampled. Default: 16.

        Returns:
            A list with n tuple entries, each representing either a negative or positive interaction pair.
            The first element on each tuple represents the sampled uid (user) and the second the sampled iid (item).
        """
        sampled_pairs = []

        while len(sampled_pairs) != n:
            null_pair = self.rng.uniform(0, self.neg_ratio + 1) > 1

            if null_pair:
                sampled_uid, sampled_iid = self.sample_negative()
            else:
                sampled_uid, sampled_iid = self.sample_positive()

            sampled_pairs.append((sampled_uid, sampled_iid))

        return sampled_pairs

    def sample_one(self):
        """Sample one positive or negative interaction pair according to the set definitions of the PointSampler
        instance.

        Returns:
            A tuple representing either a negative or positive interaction pair.
            The first element on each tuple represents the sampled uid (user) and the second the sampled iid (item).
        """
        return self.sample(n=1)[0]

    def sample_negative(self):
        """Sample one negative interaction pair.

        Returns:
            A tuple representing a negative interaction pair.
            The first element on each tuple represents the sampled uid (user) and the second the sampled iid (item).
        """
        sampled_uid, sampled_iid = next(self.null_pair_gen, (None, None))
        return sampled_uid, sampled_iid

    def sample_positive(self):
        """Sample one positive interaction pair.

        Returns:
            A tuple representing a positive interaction pair.
            The first element on each tuple represents the sampled uid (user) and the second the sampled iid (item).
        """
        sampled_interaction = next(self.pos_pair_gen, None)
        if sampled_interaction is None:
            return None, None
        sampled_uid, sampled_iid = sampled_interaction['uid'], sampled_interaction['iid']
        return sampled_uid, sampled_iid
