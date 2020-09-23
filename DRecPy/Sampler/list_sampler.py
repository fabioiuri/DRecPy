from DRecPy.Dataset import InteractionDatasetABC
import random


class ListSampler:
    """ListSampler class that abstracts sampling a list/sequence of related records.

    Args:
        interaction_dataset: An instance of the InteractionDataset type, representing the data set where points
            are being sampled from.
        group_columns: A list containing the names of the columns to be used when grouping distinct records.
        neg_ratio: The number of negative records to be sampled for each target record. Is not used if n_targets=None.
            Default: 3.
        n_targets: An optional integer defining the number of target records in each group sequence. If None is
            provided, no negative records will be sampled. Default: 5.
        negative_ids_col: A string representing the name of the column to use for sampling negative identifiers. This is
            only used if n_targets is specified. Default: 'iid'.
        interaction_threshold: An optional integer that is used as the boundary interaction value between positive and
            negative records. All values above or equal to interaction_threshold are considered positive,
            and all values bellow are considered negative. Also, all the records not present on the group selection are
            considered negative (with respect to that specific group). Default: None.
        sort_column: The name of the column to use when sorting the sampled sequence of records. Since this parameter is
            optional, when none is provided the sampled records will be sorted by record id (order they appear in the
            original file). Default: None.
        min_positive_records: The min. number of positive records required to have in each group sequence. Default: 8.
        max_positive_records: Optional number representing the max. number of positive records to have in each group
            sequence. Default: None.
        seed: An optional integer to be used as the seed value for the pseudo-random number generated used to sample
            interaction pairs. Default: None.
    """
    max_consecutive_tries = 20

    def __init__(self, interaction_dataset, group_columns, neg_ratio=3, n_targets=5, negative_ids_col='iid',
                 interaction_threshold=None, sort_column=None, min_positive_records=8, max_positive_records=None,
                 seed=None):
        assert interaction_dataset is not None, 'An interaction dataset instance is required.'

        assert InteractionDatasetABC in interaction_dataset.__class__.__bases__, f'Provided interaction_dataset  ' \
            f'argument is not subclass of InteractionDataset (found type {type(interaction_dataset)}).'

        assert interaction_dataset.has_internal_ids, \
            'The provided interaction dataset instance does not have internal ids assigned.'

        if n_targets is not None:
            assert neg_ratio is not None, 'A neg_ratio value is required.'

            assert n_targets > 0, f'The number of target records per group sequence ({n_targets}) is not valid: ' \
                                  f'should be None or a positive integer.'

            assert negative_ids_col in interaction_dataset.columns, \
                f'The negative_ids_col ({negative_ids_col}) used to sample negative ids per sequence does not exist.'

        if sort_column is not None:
            assert sort_column in interaction_dataset.columns, \
                f'The provided sort column ({sort_column}) is not present on the dataset columns ({interaction_dataset.columns}).'

        if max_positive_records is not None:
            assert max_positive_records >= min_positive_records, \
                f'The max_positive_records ({max_positive_records}) must be >= min_positive_records ({min_positive_records}).'

        self.interaction_dataset = interaction_dataset
        self.group_columns = group_columns
        self.neg_ratio = neg_ratio
        self.n_targets = n_targets
        self.interaction_threshold = interaction_threshold
        self.sort_column = sort_column
        self.min_positive_records = min_positive_records
        self.max_positive_records = max_positive_records
        self.rng = random.Random(seed)
        self.unique_groups = self.interaction_dataset.unique(group_columns).values_list(group_columns, to_list=True)
        self.negative_ids_col = negative_ids_col
        self.unique_negative_ids = set(self.interaction_dataset.unique(negative_ids_col).values_list(negative_ids_col, to_list=True))

    def sample_group_records(self, n=16):
        """Samples a list with a sequence of records that are grouped and sorted according to the instance parameters.

        Args:
            n: An integer representing the number of sets of grouped records. Default: 16.

        Returns:
            If self.n_targets is None, returns a list containing in each element a sequence of positive related records.
            If self.n_targets is not None, returns a list containing in each element a (positive_sequences,
            target_sequences, negative_ids) triple.
            Each element on the sequence of related records (positive or target) is a dict representing each record
            (contains all the row properties like 'user', 'item', 'interaction', etc.).
        """
        group_records_list = []

        for i in range(n):
            consecutive_tries = 0
            while True:
                consecutive_tries += 1

                chosen_group = self.rng.choice(self.unique_groups)
                if not isinstance(chosen_group, list):
                    chosen_group = [chosen_group]
                query = ','.join([f'{group_col} == {chosen_col_group}' for group_col, chosen_col_group
                                  in zip(self.group_columns, chosen_group)])
                group_records_ds = self.interaction_dataset.select(query)
                if self.interaction_threshold is None:
                    positive_group_records_ds = group_records_ds
                else:
                    positive_group_records_ds = group_records_ds.select(f'interaction >= {self.interaction_threshold}')

                if len(positive_group_records_ds) < self.min_positive_records or \
                        (self.n_targets is not None and len(positive_group_records_ds) < self.min_positive_records + self.n_targets):
                    if consecutive_tries > self.max_consecutive_tries:
                        raise Exception(f'Failed to sample group records, max consecutive tries reached '
                                        f'({self.max_consecutive_tries}): consider reducing the min_group_records '
                                        f'({self.min_positive_records}).')
                    continue

                positive_group_records = positive_group_records_ds.values_list()
                if self.sort_column is not None:
                    positive_group_records.sort(key=lambda x: x[self.sort_column])
                all_positive_group_records = positive_group_records

                padding = None
                if self.max_positive_records is not None and len(positive_group_records) > self.max_positive_records:
                    if self.n_targets is None:
                        padding = self.rng.randint(0, len(positive_group_records) - self.max_positive_records)
                    else:
                        padding = self.rng.randint(0, len(positive_group_records) - self.max_positive_records - self.n_targets)
                    positive_group_records = positive_group_records[padding:padding + self.max_positive_records]

                if self.n_targets is None:
                    group_records_list.append(positive_group_records)
                    break

                eligible_negative_ids = self.unique_negative_ids.difference(set([rec[self.negative_ids_col] for rec in all_positive_group_records]))
                if padding is None:
                    target_group_records = positive_group_records[self.n_targets:]
                    positive_group_records = positive_group_records[:self.n_targets]
                else:
                    target_group_records = all_positive_group_records[padding + self.max_positive_records:padding + self.max_positive_records + self.n_targets]

                # sample negative ids or skip if not possible
                n_negative_ids = self.neg_ratio * len(target_group_records)

                if len(eligible_negative_ids) < n_negative_ids:
                    if consecutive_tries > self.max_consecutive_tries:
                        raise Exception(f'Failed to sample group records, max consecutive tries reached '
                                        f'({self.max_consecutive_tries}): consider reducing the neg_ratio '
                                        f'({self.neg_ratio}) or the n_targets ({self.n_targets}).')
                    continue

                negative_ids = self.rng.sample(eligible_negative_ids, n_negative_ids)
                group_records_list.append((positive_group_records, target_group_records, negative_ids))
                break

        return group_records_list
