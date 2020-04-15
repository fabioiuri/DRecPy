from .dataset_abc import InteractionDatasetABC
from copy import deepcopy
import pandas as pd
from pandas.api.types import is_string_dtype
import random
from scipy.sparse import csr_matrix
import psutil


class MemoryInteractionDataset(InteractionDatasetABC):
    def __init__(self, path='', columns=None, **kwds):
        super().__init__(**kwds)

        self.in_memory = True
        self.columns = columns
        self.path = path

        self._users_records = None
        self._items_records = None
        self._user_mapping = None
        self._user_mapping_inv = None
        self._item_mapping = None
        self._item_mapping_inv = None
        self._cached_interaction_matrix = None  # users in rows, items in cols
        self._cached_trans_interaction_matrix = None  # items in rows, users in cols

        self._log('Trying to load data into memory...')

        if kwds.get('df') is None:
            projected_cols = [col for col in self.columns if col != 'rid']
            delimiter = kwds.get('delimiter', ',')
            has_header = kwds.get('has_header', True)

            use_cols = [i for col, i in zip(projected_cols, range(len(projected_cols))) if col is not None]
            skip_rows = 1 if has_header else 0

            self._df = pd.read_csv(path, delimiter=delimiter, names=projected_cols, encoding=kwds.get('encoding'),
                                   skiprows=skip_rows, usecols=use_cols)
            self.columns = [col for col in self.columns if col is not None]  # remove any skip cols
        else:
            self._df = kwds.get('df')

        try:
            self._df['user'] = self._df[kwds.get('user_label', 'user')]
            self._df['item'] = self._df[kwds.get('item_label', 'item')]
            self._df['interaction'] = self._df[kwds.get('interaction_label', 'interaction')]
        except Exception as e:
            raise Exception('An error occurred when converting the main columns. Required columns: "user", "item" and '
                            '"interaction". Make sure that the data you\'re loading does not have a header - if it does, '
                            'set the argument "has_header" to true.\n'
                            'More details: {e}'.format(e=str(e)))

        if self.columns is None:
            self.columns = list(self._df.columns) + ['rid']
            user_label = kwds.get('user_label', None)
            if user_label is not None and user_label is not 'user':
                self.columns.remove(user_label)

            item_label = kwds.get('item_label', None)
            if item_label is not None and item_label is not 'item':
                self.columns.remove(item_label)

            interaction_label = kwds.get('interaction_label', None)
            if interaction_label is not None and interaction_label is not 'interaction':
                self.columns.remove(interaction_label)

        # remove extra columns
        for col in self._df.columns:
            if col not in self.columns:
                self._df.drop(columns=col, inplace=True)

        # fill nan's as empty strings for str col typs
        for col in self._df.columns:
            if is_string_dtype(self._df[col]):
                self._df[col].fillna('', inplace=True)

        self._log('Done!')

    def __len__(self):
        return self._df.shape[0]

    def select(self, query, copy=True):
        new_ds = self.__copy__() if copy else self
        new_ds._apply_query(query)

        return new_ds

    def select_random_generator(self, query=None, seed=None):
        assert self.has_internal_ids is True, 'No internal ids assigned yet.'
        assert len(self) > 0, 'No records were found.'

        if query is None:
            df = self._df
        else:
            df = self._apply_query(query, in_place=False)

        assert df.shape[0] > 0, 'No records were found after applying the given query.'

        rng = random
        if seed is not None: rng = random.Random(seed)

        max_uid = self.max('uid')
        df_columns = [col for col in self.columns if col != 'rid']
        while True:
            random_uid = rng.randint(0, max_uid)
            tmp_df = df.query(f'uid == {random_uid}')
            if len(tmp_df) == 0: continue

            chosen_idx = rng.randint(0, tmp_df.shape[0] - 1)
            record = {k: v for k, v in zip(df_columns, tmp_df.iloc[chosen_idx, :])}
            record['rid'] = tmp_df.index[chosen_idx]
            yield record

    def null_interaction_pair_generator(self, interaction_threshold=None, seed=None):
        assert self.has_internal_ids is True, 'No internal ids assigned yet.'
        assert len(self) > 0, 'No records were found.'

        rng = random
        if seed is not None: rng = random.Random(seed)

        try:
            existing_null_pair_gen = self.select_random_generator(query=f'interaction <= {interaction_threshold}',
                                                                  seed=seed)
        except:
            existing_null_pair_gen = None
        finally:
            if interaction_threshold is None:
                existing_null_pair_gen = None

        max_uid = self.max('uid')
        max_iid = self.max('iid')

        if self._users_records is None:
            uid_values = self._df['uid'].values
            iid_values = self._df['iid'].values
            while True:
                if existing_null_pair_gen is not None and rng.randint(0, 5) == 0:
                    existing_null_pair = next(existing_null_pair_gen)
                    yield existing_null_pair['uid'], existing_null_pair['iid']
                else:
                    random_uid = rng.randint(0, max_uid)
                    random_iid = rng.randint(0, max_iid)
                    found = self._df[(uid_values == random_uid) & (iid_values == random_iid)].shape[0] > 0
                    if not found:
                        yield random_uid, random_iid
        else:
            while True:
                if existing_null_pair_gen is not None and rng.randint(0, 5) == 0:
                    existing_null_pair = next(existing_null_pair_gen)
                    yield existing_null_pair['uid'], existing_null_pair['iid']
                else:
                    random_uid = rng.randint(0, max_uid)
                    random_iid = rng.randint(0, max_iid)
                    if len(self._users_records[random_uid]) > len(self._items_records[random_iid]):
                        found = any(filter(lambda record: record['uid'] == random_uid, self._items_records[random_iid]))
                    else:
                        found = any(filter(lambda record: record['iid'] == random_iid, self._users_records[random_uid]))
                    if not found: yield random_uid, random_iid

    def select_user_interaction_vec(self, uid):
        assert self.has_internal_ids is True, 'Cannot retrieve user interaction vector without assigned internal ids.'
        assert self.uid_to_user(uid) is not None, f'User internal id {uid} was not found.'

        try:
            if self._cached_interaction_matrix is not None and self._cached_interaction_matrix.shape[0] <= 1:
                raise MemoryError

            if self._cached_interaction_matrix is None:
                self._build_interaction_matrix()

            return self._cached_interaction_matrix[uid]
        except MemoryError:
            # don't try to use this again (if it didn't fit in memory now, probably wont fit later)
            self._cached_interaction_matrix = csr_matrix(0)
            cols, interactions = [], []
            if self._users_records is None:
                df_segment = self._df[self._df['uid'].values == uid]
                projected_cols = [col for col in self.columns if col != 'rid']
                for row in df_segment[projected_cols].values:
                    cols.append(row[projected_cols.index('iid')])
                    interactions.append(float(row[projected_cols.index('interaction')])) # floats due to tf not supporting NPY_INT during np.array to tensor

                return csr_matrix((interactions, ([0] * len(interactions), cols)),
                                  shape=(1, self._df.drop_duplicates(subset='iid').shape[0]))[0]
            else:
                for row in self._users_records[uid]:
                    cols.append(row['iid'])
                    interactions.append(float(row['interaction'])) # floats due to tf not supporting NPY_INT during np.array to tensor

                return csr_matrix((interactions, ([0] * len(interactions), cols)), shape=(1, len(self._items_records)))[0]

    def select_item_interaction_vec(self, iid):
        assert self.has_internal_ids is True, 'Cannot retrieve user interaction vector without assigned internal ids.'
        assert self.iid_to_item(iid) is not None, f'Item internal id {iid} was not found.'

        try:
            if self._cached_interaction_matrix is not None and self._cached_interaction_matrix.shape[0] <= 1:
                raise MemoryError

            if self._cached_interaction_matrix is None:
                self._build_interaction_matrix()

            return self._cached_trans_interaction_matrix[iid]
        except MemoryError:
            # don't try to use this again (if it didn't fit in memory now, probably wont fit later)
            self._cached_interaction_matrix = csr_matrix(0)

            cols, interactions = [], []
            if self._items_records is None:
                df_segment = self._df[self._df['iid'].values == iid]
                projected_cols = [col for col in self.columns if col != 'rid']
                for row in df_segment[projected_cols].values:
                    cols.append(row[projected_cols.index('iid')])
                    interactions.append(float(row[projected_cols.index('interaction')])) # floats due to tf not supporting NPY_INT during np.array to tensor

                return csr_matrix((interactions, ([0] * len(interactions), cols)),
                                  shape=(1, self._df.drop_duplicates(subset='uid').shape[0]))[0]
            else:
                for row in self._items_records[iid]:
                    cols.append(row['uid'])
                    interactions.append(float(row['interaction'])) # floats due to tf not supporting NPY_INT during np.array to tensor

                return csr_matrix((interactions, ([0] * len(interactions), cols)), shape=(1, len(self._users_records)))[0]

    def unique(self, columns=None, copy=True):
        columns = self._handle_columns(columns)
        new_ds = self.__copy__() if copy else self

        if 'rid' not in columns: columns.append('rid')
        df_columns = [col for col in columns if col != 'rid']
        new_ds._df = new_ds._df.drop_duplicates(subset=df_columns)
        new_ds._df = new_ds._df[df_columns]
        new_ds.columns = columns

        return new_ds

    def max(self, column=None):
        self._validate_column(column)
        if column == 'rid': return self._df.index.max()
        return self._df[column].max()

    def min(self, column=None):
        self._validate_column(column)
        if column == 'rid': return self._df.index.min()
        return self._df[column].min()

    def values(self, columns=None, to_list=False):
        columns = self._handle_columns(columns)

        if self._users_records is None or not self.has_internal_ids:
            df_columns = [col for col in self.columns if col != 'rid']
            for row, rid in zip(self._df[df_columns].values, self._df.index):
                yield self._from_row_to_record(row, rid, columns, df_columns, to_list)
        else:
            for user in self._users_records:
                for record in self._users_records[user]:
                    if to_list:
                        list_interaction = [record[col] for col in columns]
                        if len(list_interaction) == 1:
                            yield list_interaction[0]
                        else:
                            yield list_interaction
                    else:
                        yield record

    def drop(self, record_ids, copy=True, keep=False):
        new_ds = self.__copy__() if copy else self

        if keep:
            new_ds._df = new_ds._df[new_ds._df.index.isin(record_ids)]
        else:
            new_ds._df = new_ds._df[~new_ds._df.index.isin(record_ids)]
        return new_ds

    def user_to_uid(self, user):
        assert self.has_internal_ids is True, 'No internal ids assigned yet.'

        user_dtype = str(self._df['user'].dtype)
        if 'int' in user_dtype:
            try:
                user = int(user)
            except ValueError:
                raise Exception(f'The provided user type does not match the inferred type (expected: int, found: {type(user)}')
        else: user = str(user)
        return self._user_mapping[user] if user in self._user_mapping else None

    def uid_to_user(self, uid):
        assert self.has_internal_ids is True, 'No internal ids assigned yet.'

        return self._user_mapping_inv[uid] if uid in self._user_mapping_inv else None

    def item_to_iid(self, item):
        assert self.has_internal_ids is True, 'No internal ids assigned yet.'

        item_dtype = str(self._df['item'].dtype)
        if 'int' in item_dtype:
            try:
                item = int(item)
            except ValueError:
                raise Exception(f'The provided item type does not match the inferred type (expected: int, found: {type(item)}')
        else: item = str(item)
        return self._item_mapping[item] if item in self._item_mapping else None

    def iid_to_item(self, iid):
        assert self.has_internal_ids is True, 'No internal ids assigned yet.'

        return self._item_mapping_inv[iid] if iid in self._item_mapping_inv else None

    def save(self, path='', columns=None, write_header=False):
        if len(path) == 0 and len(self.path) == 0: raise Exception('No save path was specified.')

        if len(path) == 0: path = self.path

        projected_cols = deepcopy(self.columns)
        if 'rid' in projected_cols: projected_cols.remove('rid')
        if 'uid' in projected_cols: projected_cols.remove('uid')
        if 'iid' in projected_cols: projected_cols.remove('iid')
        self._df[projected_cols].to_csv(path, index=None, header=write_header)

    def assign_internal_ids(self):
        self._df = self._df.copy()  # to avoid issues with other instances modifying the df (df objects are shared)

        if 'uid' not in self.columns: self.columns.append('uid')
        # convert raw user ids (user column) to internal user ids (uid)
        unique_users = self._df['user'].unique()
        user_cat = pd.Categorical(self._df['user'], categories=unique_users)
        self._df['uid'] = user_cat.codes
        # to speed up user lookups
        self._user_mapping = dict([(cat, code) for code, cat in enumerate(user_cat.categories)])
        self._user_mapping_inv = dict([(code, cat) for code, cat in enumerate(user_cat.categories)])

        if 'iid' not in self.columns: self.columns.append('iid')
        # convert raw item ids (item column) to internal user ids (iid)
        unique_items = self._df['item'].unique()
        item_cat = pd.Categorical(self._df['item'], categories=unique_items)
        self._df['iid'] = item_cat.codes
        # to speed up item lookups
        self._item_mapping = dict([(cat, code) for code, cat in enumerate(item_cat.categories)])
        self._item_mapping_inv = dict([(code, cat) for code, cat in enumerate(item_cat.categories)])

        self.has_internal_ids = True
        self._build_optimizations()

    def remove_internal_ids(self):
        self.has_internal_ids = False
        if 'uid' in self.columns:
            self._df = self._df.drop(columns=['uid', 'iid'])
            self.columns.remove('uid')
            self.columns.remove('iid')

    def apply(self, column, function):
        self._validate_column(column)
        read_only_columns = ['rid', 'uid', 'iid', 'user', 'item']
        supported_types = [int, float, str]
        if column in read_only_columns: raise Exception(f'Column "{column}" is read-only.')

        try:
            test_record_col = self._df[column].values[0]
            new_col_type = type(function(test_record_col))
            if 'numpy' in str(new_col_type): new_col_type = type(test_record_col.item())
            assert new_col_type in supported_types, f'New column type "{new_col_type}" is not supported. Supported types: {supported_types}.'

            self._df = self._df.copy()
            self._df[column] = self._df[column].apply(function)

            if self._users_records is not None:
                self._users_records = None
                self._items_records = None
                self._build_optimizations()  # rebuild them

            if column == 'interaction' and self._cached_interaction_matrix is not None:
                self._cached_interaction_matrix = None
                self._cached_trans_interaction_matrix = None
                self._build_interaction_matrix()

        except Exception as e:
            raise Exception(f'Failed to apply operation on column "{column}". Details: {e}')

    def __str__(self):
        return f'[MemoryInteractionDataset with shape {(len(self),  len(self.columns))}]'

    def __copy__(self):
        new = MemoryInteractionDataset.__new__(MemoryInteractionDataset)
        new.in_memory = self.in_memory
        new.verbose = self.verbose
        new.path = self.path
        new.has_internal_ids = self.has_internal_ids
        new.columns = deepcopy(self.columns)

        new._df = self._df  # df objects are shared and whenever a modification is done, it should be cloned
        new._user_mapping = self._user_mapping
        new._user_mapping_inv = self._user_mapping_inv
        new._item_mapping = self._item_mapping
        new._item_mapping_inv = self._item_mapping_inv

        new._users_records = None
        new._items_records = None
        new._cached_interaction_matrix = None
        new._cached_trans_interaction_matrix = None

        return new

    """ Auxiliary methods """
    def _apply_query(self, query, in_place=True):
        query = str(query)
        query_segments = query.split(',')

        condition_vector = None

        for segment in query_segments:
            segment = segment.strip()
            try:
                column, operator, value = segment.split(" ")
            except ValueError:
                raise Exception(
                    f'Query segment failed to be parsed. Check if there are no missing spaces or invalid characters.'
                    f' Query segment: "{segment}"'
                )

            assert column is not None or operator is not None or value is not None, \
                'Invalid get query, must be in form: column1 operator1 value1, column2 operator2 value2, ...'
            assert column in self.columns, f'Unexpected column "{column}".'

            if operator not in ['>', '>=', '<=', '<', '==', '!=']:
                raise Exception(f'Unexpected operator "{operator}".')

            value = eval(value)
            if column == 'rid':
                col_dtype = str(self._df.index.dtype)
            else:
                col_dtype = str(self._df[column].dtype)

            # use internal ids aux structures to speed up search
            if column == 'user' and self.has_internal_ids and 'int' not in col_dtype:
                column = 'uid'
                value = self.user_to_uid(value)
            elif column == 'item' and self.has_internal_ids and 'int' not in col_dtype:
                column = 'iid'
                value = self.item_to_iid(value)
            else:
                try:
                    if 'int' in col_dtype:
                        if type(value) != float and type(value) != int:
                            raise ValueError
                    elif 'float' in col_dtype:
                        value = float(value)
                    elif type(value) is not list:
                        value = str(value)
                except ValueError:
                    raise Exception(
                        f'Query "{query}" was failed to be parsed: check if no invalid comparisons are being made '
                        f'(column of type int being compared to a str, or vice versa).'
                    )

            if column == 'rid':
                values = self._df.index.values
            else:
                values = self._df[column].values

            if operator == '>':
                if condition_vector is None:
                    condition_vector = (values > value)
                else:
                    condition_vector &= (values > value)
            elif operator == '>=':
                if condition_vector is None:
                    condition_vector = (values >= value)
                else:
                    condition_vector &= (values >= value)
            elif operator == '<=':
                if condition_vector is None:
                    condition_vector = (values <= value)
                else:
                    condition_vector &= (values <= value)
            elif operator == '<':
                if condition_vector is None:
                    condition_vector = (values < value)
                else:
                    condition_vector &= (values < value)
            elif operator == '==':
                if condition_vector is None:
                    condition_vector = (values == value)
                else:
                    condition_vector &= (values == value)
            elif operator == '!=':
                if condition_vector is None:
                    condition_vector = (values != value)
                else:
                    condition_vector &= (values != value)

        if in_place:
            self._df = self._df[condition_vector]
            return self._df
        else:
            return self._df[condition_vector]

    def _from_row_to_record(self, row, rid, projected_columns, df_columns, to_list):
        record = dict()

        for col, value in zip(df_columns, row):
            if col not in projected_columns: continue
            dtype = str(self._df[col].dtype)

            if 'int' in dtype:
                record[col] = int(value)
            elif 'float' in dtype:
                record[col] = float(value)
            else:
                record[col] = value

        if 'rid' in projected_columns: record['rid'] = rid
        if to_list:
            list_interaction = [record[col] for col in projected_columns]
            if len(list_interaction) == 1: return list_interaction[0]
            else: return list_interaction
        return record

    def _build_optimizations(self):
        assert self.has_internal_ids is True, 'No internal ids assigned yet.'

        df_mem_usage = self._df.memory_usage(deep=True).sum()
        free_mem = psutil.virtual_memory().free
        estimated_mem_after = df_mem_usage * 2.2

        if free_mem <= estimated_mem_after:
            self._log('Not enough free memory: skip state optimization')
            return

        self._log('Optimizing state...')

        users_records = dict()
        items_records = dict()

        for row in self.values():
            if row['uid'] not in users_records:
                users_records[row['uid']] = []
            users_records[row['uid']].append(row)

            if row['iid'] not in items_records:
                items_records[row['iid']] = []
            items_records[row['iid']].append(row)

        self._users_records = users_records
        self._items_records = items_records

    def _build_interaction_matrix(self):
        assert self.has_internal_ids is True, 'Cannot retrieve user interaction vector without assigned internal ids.'

        users, cols, interactions = [], [], []
        if self._users_records is None:
            unique_uid_df = self._df.drop_duplicates(subset='uid')[['uid']]
            for uid in unique_uid_df.values:
                uid = uid[0]
                df_segment = self._df[self._df['uid'].values == uid]
                projected_cols = [col for col in self.columns if col != 'rid']
                for row in df_segment[projected_cols].values:
                    users.append(uid)
                    cols.append(row[projected_cols.index('iid')])
                    interactions.append(float(row[projected_cols.index('interaction')]))  # floats due to tf not supporting NPY_INT during np.array to tensor

            self._cached_interaction_matrix = csr_matrix((interactions, (users, cols)),
                                                         shape=(unique_uid_df.shape[0],
                                                                self._df.drop_duplicates(subset='iid').shape[0]))
        else:
            for uid in self._users_records:
                for row in self._users_records[uid]:
                    users.append(uid)
                    cols.append(row['iid'])
                    interactions.append(float(row['interaction']))  # floats due to tf not supporting NPY_INT during np.array to tensor

            self._cached_interaction_matrix = csr_matrix((interactions, (users, cols)),
                                                         shape=(len(self._users_records), len(self._items_records)))

        self._cached_trans_interaction_matrix = self._cached_interaction_matrix.transpose()


