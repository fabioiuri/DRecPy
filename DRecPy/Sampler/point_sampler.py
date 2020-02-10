from DRecPy.Dataset import InteractionDatasetABC
import random


class PointSampler:
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

        self.null_pair_gen = self.interaction_dataset.null_interaction_pair_generator(seed=seed)
        if self.interaction_threshold is None:
            self.pos_pair_gen = self.interaction_dataset.select_random_generator(seed=seed)
        else:
            self.pos_pair_gen = self.interaction_dataset.select_random_generator(f'interaction >= {interaction_threshold}',
                                                                                 seed=seed)

    def sample(self, n=16):
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
        return self.sample(n=1)[0]

    def sample_negative(self):
        sampled_uid, sampled_iid = next(self.null_pair_gen)  # todo: behaviour when empty?
        return sampled_uid, sampled_iid

    def sample_positive(self):
        sampled_interaction = next(self.pos_pair_gen)  # todo: behaviour when empty?
        sampled_uid, sampled_iid = sampled_interaction['uid'], sampled_interaction['iid']
        return sampled_uid, sampled_iid
