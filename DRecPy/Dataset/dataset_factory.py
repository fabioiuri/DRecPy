from .db_dataset import DatabaseInteractionDataset
from .mem_dataset import MemoryInteractionDataset


class InteractionsDatasetFactory:
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
        return MemoryInteractionDataset(df=df.copy(), user_label=user_label, item_label=item_label,
                                        interaction_label=interaction_label, **kwds)
