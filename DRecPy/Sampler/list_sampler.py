from DRecPy.Dataset import InteractionDatasetABC
import random


class ListSampler:
    """ListSampler class that abstracts sampling a set of related records.

    Args:
        interaction_dataset: An instance of the InteractionDataset type, representing the data set where points
            are being sampled from.
        group_columns: A list containing the names of the columns to be used when grouping distinct records.
        sort_column: The name of the column to use when sorting the sampled set of records. Since this parameter is
            optional, when none is provided the sampled records will be sorted by record id (order they appear in the
            original file). Default: None.
        seed: An optional integer to be used as the seed value for the pseudo-random number generated used to sample
            interaction pairs. Default: None.
    """
    def __init__(self, interaction_dataset, group_columns, sort_column=None, seed=None):
        assert interaction_dataset is not None, 'An interaction dataset instance is required.'

        assert InteractionDatasetABC in interaction_dataset.__class__.__bases__, f'Provided interaction_dataset  ' \
            f'argument is not subclass of InteractionDataset (found type {type(interaction_dataset)}).'

        assert interaction_dataset.has_internal_ids, \
            'The provided interaction dataset instance does not have internal ids assigned.'

        self.interaction_dataset = interaction_dataset
        self.group_columns = group_columns
        self.sort_column = sort_column
        self.rng = random.Random(seed)
        self.unique_groups = self.interaction_dataset.unique(group_columns).values_list(group_columns, to_list=True)

    def sample_group_records(self, n=16, min_group_records=8, max_group_records=None):
        """Samples a list with a set of records that are grouped and sorted according to the instance parameters.

        Args:
            n: An integer representing the number of sets of grouped records. Default: 16.
            min_group_records: The min. number of records required to have in each set. Default: 8.
            max_group_records: Optional number representing the max. number of records to have in each set.
                Default: None.

        Returns:
            A list containing in each element a set of related records. Each element on the set of related records is a
            dict representing each record (contains all the row properties like 'user', 'item', 'interaction', etc.).
            If max_group_records is not defined, the returned list has n elements, where each element has at least
            min_group_records records.
        """
        group_records_list = []

        for i in range(n):
            while True:
                chosen_group = self.rng.choice(self.unique_groups)
                if not isinstance(chosen_group, list):
                    chosen_group = [chosen_group]
                query = ','.join([f'{group_col} == {chosen_col_group}' for group_col, chosen_col_group
                                  in zip(self.group_columns, chosen_group)])
                group_records_ds = self.interaction_dataset.select(query)

                if len(group_records_ds) >= min_group_records:
                    break

            group_records = group_records_ds.values_list()
            if self.sort_column is not None:
                group_records.sort(key=lambda x: x[self.sort_column])

            if max_group_records is None or len(group_records) <= max_group_records:
                group_records_list.append(group_records[:max_group_records])

            padding = self.rng.randint(0, len(group_records) - max_group_records)
            group_records_list.append(group_records[padding:max_group_records + padding])

        return group_records_list



