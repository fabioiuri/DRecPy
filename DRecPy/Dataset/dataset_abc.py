from abc import abstractmethod, ABC
from copy import copy


class InteractionDatasetABC(ABC):

    def __init__(self, **kwds):
        self.verbose = kwds.get('verbose', True)
        self.has_internal_ids = False
        self.columns = None

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def select(self, query, copy=True):
        pass

    @abstractmethod
    def select_random_generator(self, query=None):
        pass

    @abstractmethod
    def null_interaction_pair_generator(self):
        pass

    def select_one(self, query, columns=None, to_list=False):
        return next(self.select(query).values(columns=None, to_list=False), None)

    @abstractmethod
    def select_user_interaction_vec(self, uid):
        pass

    @abstractmethod
    def select_item_interaction_vec(self, iid):
        pass

    def exists(self, query):
        return False if self.select_one(query) is None else True

    @abstractmethod
    def unique(self, columns=None, copy=True):
        pass

    def count_unique(self, columns=None):
        return len(self.unique(columns))

    @abstractmethod
    def max(self, column=None):
        pass

    @abstractmethod
    def min(self, column=None):
        pass

    @abstractmethod
    def values(self, columns=None, to_list=False):
        pass

    def values_list(self, columns=None, to_list=False):
        return [record for record in self.values(columns=columns, to_list=to_list)]

    @abstractmethod
    def drop(self, record_ids, copy=True, keep=False): # todo: add tests for keep=True
        pass

    @abstractmethod
    def assign_internal_ids(self):
        pass

    @abstractmethod
    def remove_internal_ids(self):
        pass

    @abstractmethod
    def user_to_uid(self, user):
        pass

    @abstractmethod
    def uid_to_user(self, uid):
        pass

    @abstractmethod
    def item_to_iid(self, item):
        pass

    @abstractmethod
    def apply(self, column, function):
        pass

    @abstractmethod
    def iid_to_item(self, iid):
        pass

    @abstractmethod
    def save(self, path, columns=None, write_header=False):
        pass

    @abstractmethod
    def __copy__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()

    """ Private methods """
    def _validate_column(self, column):
        assert column is not None, 'No column was given.'
        assert type(column) is str, f'Unexpected column type "{type(column)}".'
        assert column in self.columns, f'Unexpected column "{column}".'

    def _handle_columns(self, columns):
        if columns is None: columns = copy(self.columns)
        if type(columns) is not list: columns = [columns]

        for c in columns:
            assert c in self.columns, f'Unexpected column "{c}".'

        return columns

    def _log(self, msg):
        if not self.verbose: return
        print(f'[{self.__class__.__name__}] {msg}')


