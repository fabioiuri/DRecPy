from .db_dataset import DatabaseInteractionDataset
from .mem_dataset import MemoryInteractionDataset


class InteractionsDatasetFactory:
    """InteractionsDatasetFactory creates InteractionDataset instances.

    Args:
        path: A string representing the path to the file where the dataset is located at.
        columns: A list with the names of the columns present on the dataset, ordered accordingly to the column order
            present in the dataset file. Required column names: 'user', 'item', 'interaction'.
        delimiter: A string representing the delimiter used on the dataset file. Default: ','.
        has_header: A boolean indicating whether the dataset file has a header row or not (skip first row or not?).
            Default: false.
        in_memory: A boolean indicating whether to load the dataset: in memory or out of memory. Default: True.
        verbose: A boolean indicating whether to log info messages or not. Default: True.
    """
    def __new__(cls, path='', columns=None, delimiter=',', has_header=False, in_memory=True, **kwds):
        is_db = path.endswith('.sqlite')

        if columns is None and not is_db: raise Exception('Missing the "columns" argument.')
        if columns is not None and 'uid' in columns: raise Exception('Cannot import column "uid".')
        if columns is not None and 'iid' in columns: raise Exception('Cannot import column "iid".')
        if columns is not None: columns = columns + ['rid']

        try:
            if in_memory and not is_db:
                dataset = MemoryInteractionDataset(path=path, columns=columns, delimiter=delimiter,
                                                   has_header=has_header, **kwds)
            else:
                dataset = DatabaseInteractionDataset(path=path, columns=columns, delimiter=delimiter,
                                                     has_header=has_header, **kwds)
        except MemoryError as e:
            if not in_memory: raise e
            dataset = DatabaseInteractionDataset(path=path, columns=columns, delimiter=delimiter,
                                                 has_header=has_header, **kwds)

        return dataset

    @staticmethod
    def read_df(df, user_label='user', item_label='item', interaction_label='interaction', **kwds):
        """Convert the provided dataframe into a InteractionDataset instance.

        Args:
            df: A dataframe containing the dataset to be imported.
            user_label: The name of the column containing the user identifiers. Default: 'user'.
            item_label: The name of the column containing the item identifiers. Default: 'item'.
            interaction_label: The name of the column containing the interaction values. Default: 'interaction'.

        Returns:
            A InteractionDataset instance containing the provided data.
        """
        return MemoryInteractionDataset(df=df.copy(), user_label=user_label, item_label=item_label,
                                        interaction_label=interaction_label, **kwds)
