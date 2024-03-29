from .dataset_abc import InteractionDatasetABC
from .file_utils import data_path
from .file_utils import register_temp_file
from .file_utils import unregister_temp_file
import sqlite3 as sql
from copy import copy as shallowcopy
import csv
import random
from tqdm import tqdm
from os import remove
from os import rename
from os.path import isfile
from os.path import join
from os.path import expanduser
from scipy.sparse import csr_matrix


class DatabaseInteractionDataset(InteractionDatasetABC):
    _open_value_generators = set()  # a set containing the still open value generators (not fully read or not GCed)
    _shared_db_instances = dict()  # maps the db file paths to a list of object ids currently using it
    _shared_db_table_instances = dict()  # maps the db file path + table name to a list of object ids currently using it
    _opt_ratio_threshold = 0.8  # after at least a 20% reduction between old and new states
    _opt_n_query_threshold = 3  # only check if should optimize after X queries w/o optimize
    _opt_n_rows_threshold = 5000  # only optimize if the row diff between states is at least X
    _opt2_n_direct_interactions_threshold = 4  # another optimization strategy independend from the previous one: optimize if the #direct interactions is bigger than this
    _default_user_vec_cache_limit = 1000
    _default_item_vec_cache_limit = 1000

    def __init__(self, path='', columns=None, **kwds):
        super().__init__(**kwds)

        self.in_memory = False
        self.columns = columns
        self.chunk_size = kwds.get('chunksize', 10 ** 5)

        self._state_query = ''
        self._queries_wo_optimize = 0
        self._conn = None
        self._db_path = None
        self._active_table = kwds.get('active_table', 'interactions')
        self._col_types = None
        self._n_rows = None
        self._n_direct_interactions = 0

        self._user_interaction_vec_cache = dict()
        self._item_interaction_vec_cache = dict()
        self._user_vec_cache_limit = self._default_user_vec_cache_limit
        self._item_vec_cache_limit = self._default_item_vec_cache_limit

        if kwds.get('user_vec_cache_limit', None) is not None:
            self._user_vec_cache_limit = kwds.get('user_vec_cache_limit')
        if kwds.get('item_vec_cache_limit', None) is not None:
            self._item_vec_cache_limit = kwds.get('item_vec_cache_limit')

        path = expanduser(path)
        if path.endswith('.sqlite'):
            self._load_from_db(path)
        else:
            self._load_from_csv(path, kwds.get('delimiter', ','), kwds.get('has_header', True), kwds.get('encoding', None))

        if self._db_path not in self._shared_db_instances:
            self._shared_db_instances[self._db_path] = set()
        self._shared_db_instances[self._db_path].add(id(self))

        if self._db_path + self._active_table not in self._shared_db_table_instances:
            self._shared_db_table_instances[self._db_path + self._active_table] = set()
        self._shared_db_table_instances[self._db_path + self._active_table].add(id(self))

        self._n_rows = len(self)
        self._n_rows_before_opt = self._n_rows

        self._log('Done!')

    def __len__(self):
        if self._n_rows is None:
            c = self._open_cursor()
            c.execute(f'SELECT COUNT(*) FROM ({self._curr_state_source()})')
            self._n_rows = c.fetchone()[0]
        return self._n_rows

    def select(self, query, copy=True):
        new_ds = self.copy() if copy else self
        if copy: self._record_interaction()

        query_conditions = new_ds._parse_query(query)
        new_state = new_ds._build_new_state(query_conditions)
        new_ds._update_state(new_state)

        return new_ds

    def select_one(self, query, columns=None, to_list=False):
        self._record_interaction()
        query_conditions = self._parse_query(query)
        columns = self._handle_columns(columns)

        new_state = self._build_new_state(query_conditions, limit=1)

        c = self._open_cursor()

        row = c.execute(new_state).fetchone()
        if row is None: return None

        return self._from_row_to_record(row, columns, to_list)

    def select_user_interaction_vec(self, uid):
        assert self.has_internal_ids is True, 'Cannot retrieve user interaction vector without assigned internal ids.'
        assert self.uid_to_user(uid) is not None, f'User internal id {uid} was not found.'

        if uid in self._user_interaction_vec_cache:
            return self._user_interaction_vec_cache[uid]

        if len(self._user_interaction_vec_cache) > self._user_vec_cache_limit:
            self._user_interaction_vec_cache.clear()

        cols, interactions = [], []
        max_iid = self.max('iid')
        user_interactions = self.select(f'uid == {uid}').values(columns=['iid', 'interaction'], to_list=True)

        for iid, interaction in user_interactions:
            cols.append(iid)
            interactions.append(float(interaction))  # floats due to tf not supporting NPY_INT during np.array to tensor

        try:
            self._user_interaction_vec_cache[uid] = csr_matrix((interactions, ([0] * len(interactions), cols)),
                                                               shape=(1, max_iid + 1))[0]
        except MemoryError:
            self._user_interaction_vec_cache.clear()

        return self._user_interaction_vec_cache[uid]

    def select_item_interaction_vec(self, iid):
        assert self.has_internal_ids is True, 'Cannot retrieve user interaction vector without assigned internal ids.'
        assert self.iid_to_item(iid) is not None, f'Item internal id {iid} was not found.'

        if iid in self._item_interaction_vec_cache:
            return self._item_interaction_vec_cache[iid]

        if len(self._item_interaction_vec_cache) > self._item_vec_cache_limit:
            self._item_interaction_vec_cache.clear()

        cols, interactions = [], []
        max_uid = self.max('uid')
        item_interactions = self.select(f'iid == {iid}').values(columns=['uid', 'interaction'], to_list=True)

        for uid, interaction in item_interactions:
            cols.append(uid)
            interactions.append(float(interaction))  # floats due to tf not supporting NPY_INT during np.array to tensor

        try:
            self._item_interaction_vec_cache[iid] = csr_matrix((interactions, ([0] * len(interactions), cols)),
                                                               shape=(1, max_uid + 1))[0]
        except MemoryError:
            self._item_interaction_vec_cache.clear()

        return self._item_interaction_vec_cache[iid]

    def select_random_generator(self, query=None, seed=None):
        assert self.has_internal_ids is True, 'No internal ids assigned yet.'

        if query is None:
            return self._select_random_generator(seed)

        new_ds = self.copy()
        query_conditions = new_ds._parse_query(query)
        new_state = new_ds._build_new_state(query_conditions)
        new_ds._update_state(new_state)
        return new_ds._select_random_generator(seed)

    def _select_random_generator(self, seed=None):
        assert len(self) > 0, 'No records were found to sample from.'

        rng = random
        if seed is not None: rng = random.Random(seed)

        c = self._open_cursor()
        max_uid = self.max('uid')
        while True:
            random_uid = rng.randint(0, max_uid)
            query_conditions = self._parse_query(f'uid == {random_uid}')
            new_state = self._build_new_state(query_conditions)
            count = c.execute(f'SELECT COUNT(*) FROM ({new_state})').fetchone()[0]
            if count == 0: continue

            chosen_record = None
            chosen_idx = rng.randint(0, count - 1)

            c.execute(new_state)
            curr_idx = 0
            for row in c:
                if curr_idx == chosen_idx:
                    chosen_record = {k: v for k, v in zip(self.columns, row)}
                    break
                curr_idx += 1

            if chosen_record is not None: yield chosen_record

    def null_interaction_pair_generator(self, interaction_threshold=None, seed=None):
        assert self.has_internal_ids is True, 'No internal ids assigned yet.'
        assert len(self) > 0, 'No records were found to sample from.'

        rng = random
        if seed is not None: rng = random.Random(seed)

        if interaction_threshold is None:
            existing_null_pair_gen = None
        else:
            try:
                negative_ds = self.copy()
                query_conditions = negative_ds._parse_query(f'interaction < {interaction_threshold}')
                new_state = negative_ds._build_new_state(query_conditions)
                negative_ds._update_state(new_state)
                existing_null_pair_gen = negative_ds._select_random_generator(seed)
                next(existing_null_pair_gen)  # force error if there's no matches
            except:
                existing_null_pair_gen = None

        max_uid = self.max('uid')
        max_iid = self.max('iid')
        n_possible_combs = max_uid * max_iid

        c = self._open_cursor()
        while True:
            if existing_null_pair_gen is not None and rng.randint(0, n_possible_combs + len(negative_ds)) >= n_possible_combs:
                existing_null_pair = next(existing_null_pair_gen)
                yield existing_null_pair['uid'], existing_null_pair['iid']
            else:
                random_uid = rng.randint(0, max_uid)
                random_iid = rng.randint(0, max_iid)
                query_conditions = self._parse_query(f'uid == {random_uid}, iid == {random_iid}')
                new_state = self._build_new_state(query_conditions)
                if c.execute(new_state).fetchone() is None:
                    yield random_uid, random_iid

    def unique(self, columns=None, copy=True):
        columns = self._handle_columns(columns)
        new_ds = self.copy() if copy else self
        new_ds.columns = columns
        if copy: self._record_interaction()

        # make sure to always maintain rid on DatabaseInteractionsDataset instances
        new_cols = ','.join(columns)
        new_state = f'SELECT {new_cols} FROM ({new_ds._curr_state_source()}) WHERE rid IN ' \
            f'(SELECT MIN(rid) FROM ({new_ds._curr_state_source()}) ' \
            f'GROUP BY {new_cols})'
        if 'rid' not in columns:
            new_state = f'SELECT {new_cols}, rid FROM ({new_ds._curr_state_source()}) WHERE rid IN ' \
                f'(SELECT MIN(rid) FROM ({new_ds._curr_state_source()}) ' \
                f'GROUP BY {new_cols})'
            new_ds.columns.append('rid')

        new_ds._update_state(new_state)
        return new_ds

    def count_unique(self, columns=None):
        self._record_interaction()
        columns = self._handle_columns(columns)

        c = self._open_cursor()
        c.execute(f'SELECT COUNT(*) FROM (SELECT DISTINCT {",".join(columns)} FROM ({self._curr_state_source()}))')
        return c.fetchone()[0]

    def max(self, column=None):
        self._record_interaction()
        self._validate_column(column)

        c = self._open_cursor()
        c.execute(f'SELECT MAX({column}) FROM ({self._curr_state_source()})')
        return c.fetchone()[0]

    def min(self, column=None):
        self._record_interaction()
        self._validate_column(column)

        c = self._open_cursor()
        c.execute(f'SELECT MIN({column}) FROM ({self._curr_state_source()})')
        return c.fetchone()[0]

    def values(self, columns=None, to_list=False):
        class ValueGenerator:
            def __init__(self, open_value_generators, row_to_record_fn):
                self.open_value_generators = open_value_generators
                self.row_to_record_fn = row_to_record_fn
                self.iter = self.__iter__()
                self.removed_gen = False
                open_value_generators.add(c)

            def __iter__(self):
                for row in c:
                    yield self.row_to_record_fn(row, columns, to_list)

                self.removed_gen = True
                self.open_value_generators.remove(c)
                c.close()

            def __next__(self):
                return next(self.iter)

            def __del__(self):
                if not self.removed_gen:
                    self.open_value_generators.remove(c)
                    self.removed_gen = True
                    try:
                        c.close()
                    except Exception:
                        pass

            def close(self):
                self.__del__()

        self._record_interaction()
        columns = self._handle_columns(columns)

        curr_state_query = self._build_new_state()
        c = self._open_cursor()
        c.execute(curr_state_query)

        return ValueGenerator(self._open_value_generators, self._from_row_to_record)

    def drop(self, record_ids, copy=True, keep=False):
        new_ds = self.copy() if copy else self
        if copy: self._record_interaction()

        rids = ','.join([str(r) for r in record_ids])
        cols = ','.join(new_ds.columns)
        if keep:
            new_state = f'SELECT {cols} FROM ({new_ds._curr_state_source()}) WHERE rid IN ({rids})'
        else:
            new_state = f'SELECT {cols} FROM ({new_ds._curr_state_source()}) WHERE rid NOT IN ({rids})'
        new_ds._update_state(new_state)
        return new_ds

    def user_to_uid(self, user):
        assert self.has_internal_ids is True, 'No internal ids assigned yet.'

        if self._col_types['user'] is int:
            try:
                user = int(user)
            except ValueError:
                raise Exception(f'The provided user type does not match the inferred type (expected: int, found: {type(user)}')
        else:
            user = str(user)

        c = self._open_cursor()
        c.execute(f'SELECT uid FROM ({self._active_table}) WHERE user = ?', (user,))
        ret = c.fetchone()

        return None if ret is None else ret[0]

    def uid_to_user(self, uid):
        assert self.has_internal_ids is True, 'No internal ids assigned yet.'

        c = self._open_cursor()
        c.execute(f'SELECT user FROM ({self._active_table}) WHERE uid = ?', (uid,))
        ret = c.fetchone()

        return None if ret is None else ret[0]

    def item_to_iid(self, item):
        assert self.has_internal_ids is True, 'No internal ids assigned yet.'

        if self._col_types['item'] is int:
            try:
                item = int(item)
            except ValueError:
                raise Exception(f'The provided item type does not match the inferred type (expected: int, found: {type(item)}')
        else:
            item = str(item)

        c = self._open_cursor()
        c.execute(f'SELECT iid FROM ({self._active_table}) WHERE item = ?', (item,))
        ret = c.fetchone()

        return None if ret is None else ret[0]

    def iid_to_item(self, iid):
        assert self.has_internal_ids is True, 'No internal ids assigned yet.'

        c = self._open_cursor()
        c.execute(f'SELECT item FROM ({self._active_table}) WHERE iid = ?', (iid,))
        ret = c.fetchone()

        return None if ret is None else ret[0]

    def save(self, path='', columns=None, write_header=False):
        """Persists the current dataset instance in the provided path, as a csv or sqlite file.
        Note that internal identifiers, such as the row id (rid), user internal id (uid) and item internal id (iid)
        are never persisted, since they're only useful during runtime.

        Args:
            path: A string that represents the path where the current dataset values will be persisted.
                If it ends in ".sqlite" then a sqlite db file will be persisted in the provided path.
                Otherwise a csv file will be persisted.
            columns: An optional list with the names of the columns that should be persisted. Default: all columns.
            write_header: A boolean indicating whether to write the csv header on the persisted file. Default: False.

        Returns:
            None.
        """
        if len(path) == 0 and '.tmp.' in self._db_path:
            raise Exception('No save path was specified.')

        if len(path) == 0: path = self._db_path

        if path.endswith('.sqlite'):
            if path == self._db_path and len(self._shared_db_instances[path]) > 1 or \
                    path != self._db_path and path in self._shared_db_instances and len(self._shared_db_instances[path]) > 0:
                raise Exception(f'Cannot save the current data into "{path}" because there are other '
                                f'instances that are using this file.')
            tmp_path = path + ".tmp"
            if isfile(tmp_path): remove(tmp_path)

            new_db = sql.connect(tmp_path, check_same_thread=False)
            self._set_pragmas(new_db)
            new_db_c = new_db.cursor()

            self._build_table(new_db_c)
            self._copy_to_new_db(new_db_c)
            self._build_indexes(new_db_c)
            new_db_c.close()
            new_db.commit()
            new_db.close()

            if path == self._db_path:  # updating origin db - requires to re-open connection
                self._conn.close()
                if isfile(path): remove(path)
                rename(tmp_path, path)
                self._open_connection(path, loading=True)
            else:  # saving into distinct destination
                if isfile(path): remove(path)
                rename(tmp_path, path)
        else:
            projected_cols = self._handle_columns(columns)
            if 'rid' in projected_cols: projected_cols.remove('rid')
            if 'uid' in projected_cols: projected_cols.remove('uid')
            if 'iid' in projected_cols: projected_cols.remove('iid')
            with open(path, 'w') as f:
                if write_header:
                    f.write(','.join(projected_cols))
                    f.write('\n')
                for row in self.values(columns=projected_cols, to_list=True):
                    f.write(','.join([str(col) if col is not None else '' for col in row]))
                    f.write('\n')

    def assign_internal_ids(self):
        self._optimize_states()
        self._conn.commit()

        # create cursor for reads and updates
        c = self._open_cursor()
        self._drop_internal_id_indexes(c, table_name=self._active_table)
        update_c = self._open_cursor()
        update_c.execute('BEGIN')

        # assigns internal ids to users (uids)
        if 'uid' not in self.columns: self.columns.append('uid')
        self._col_types['uid'] = int
        c.execute(f'SELECT COUNT(DISTINCT(user)) FROM {self._active_table}')
        n_uniq_users = c.fetchone()[0]
        c.execute(f'SELECT DISTINCT(user) FROM {self._active_table} ORDER BY rid')
        curr_uid = 0
        _iter = c
        if self.verbose: _iter = tqdm(c, total=n_uniq_users, desc='Assigning internal user ids')
        for row in _iter:
            user = row[0]
            update_c.execute(f'UPDATE {self._active_table} SET uid = ? WHERE user = ?', (curr_uid, user))
            curr_uid += 1

        # assigns internal ids to items (iids)
        if 'iid' not in self.columns: self.columns.append('iid')
        self._col_types['iid'] = int
        c.execute(f'SELECT COUNT(DISTINCT(item)) FROM {self._active_table}')
        n_uniq_items = c.fetchone()[0]
        c.execute(f'SELECT DISTINCT(item) FROM {self._active_table} ORDER BY rid')
        curr_iid = 0
        _iter = c
        if self.verbose: _iter = tqdm(c, total=n_uniq_items, desc='Assigning internal item ids')
        for row in _iter:
            item = row[0]
            update_c.execute(f'UPDATE {self._active_table} SET iid = ? WHERE item = ?', (curr_iid, item))
            curr_iid += 1

        update_c.execute("END")

        self._create_internal_id_indexes(c, table_name=self._active_table)  # after optimize_states, self._state_query = new table name
        self._conn.commit()
        self.has_internal_ids = True

    def remove_internal_ids(self):
        self.has_internal_ids = False
        if 'uid' in self.columns:
            # no need to remove from db, since it doesnt slow down - removing projected columns is enough
            self.columns.remove('uid')
            self.columns.remove('iid')

    def apply(self, column, function):
        self._validate_column(column)
        read_only_columns = ['rid', 'uid', 'iid', 'user', 'item']
        supported_types = [int, float, str]
        if column in read_only_columns: raise Exception(f'Column "{column}" is read-only.')

        try:
            value_gen = self.values(columns=[column])
            test_record = next(value_gen)
            value_gen.close()
            test_record[column] = function(test_record[column])
            new_col_type = type(test_record[column])
            assert new_col_type in supported_types, f'New column type "{new_col_type}" is not supported. Supported types: {supported_types}.'

            if new_col_type == self._col_types[column]:  # type does not change
                self._optimize_states(skip_empty_query=False)  # migrate records if current table is shared

                cur_read = self._open_cursor()
                cur_updt = self._open_cursor()
                cur_updt.execute('BEGIN')

                cur_read.execute(self._build_new_state())
                _iter = cur_read
                if self.verbose: _iter = tqdm(cur_read, total=len(self), desc='Applying transformation')
                for row in _iter:
                    record = self._from_row_to_record(row, [column, 'rid'])
                    cur_updt.execute(f'UPDATE {self._active_table} SET {column} = ? WHERE rid = ?',
                                     (function(record[column]), record['rid']))

                cur_updt.execute("END")
            else:
                # type changes, require table structure change
                cur_read = self._open_cursor()
                cur_read.execute(self._build_new_state())

                self._col_types[column] = new_col_type
                new_table_name = f'interactions_{str(random.random()).split(".")[1]}'
                c = self._open_cursor()
                self._build_table(c, new_table_name)

                cols = ','.join(self.columns)
                vars = ','.join('?' * len(self.columns))

                updt_col_index = self.columns.index(column)
                chunk = []
                _iter = cur_read
                if self.verbose: _iter = tqdm(cur_read, total=len(self), desc='Applying transformation')
                for row in _iter:
                    record = self._from_row_to_record(row, self.columns, to_list=True)
                    record[updt_col_index] = function(record[updt_col_index])
                    chunk.append(tuple(record))
                    if len(chunk) == self.chunk_size:
                        c.executemany(f'INSERT INTO {new_table_name} ({cols}) VALUES ({vars})', chunk)
                        chunk = []

                if len(chunk) > 0:
                    c.executemany(f'INSERT INTO {new_table_name} ({cols}) VALUES ({vars})', chunk)

                self._build_indexes(self._open_cursor(), table_name=new_table_name)
                self._shared_db_table_instances[self._db_path + self._active_table].remove(id(self))
                self._shared_db_table_instances[self._db_path + new_table_name] = set()
                self._shared_db_table_instances[self._db_path + new_table_name].add(id(self))
                self._active_table = new_table_name
                self._state_query = ''

            self._conn.commit()

        except Exception as e:
            raise Exception(f'Failed to apply operation on column "{column}". Details: {e}')

    def copy(self):
        new = type(self)(path=self._db_path, active_table=self._active_table, columns=shallowcopy(self.columns),
                         verbose=False)

        new.columns = shallowcopy(self.columns)
        new.has_internal_ids = self.has_internal_ids
        new._n_rows = self._n_rows
        new._state_query = shallowcopy(self._state_query)
        new._col_types = shallowcopy(self._col_types)
        new._queries_wo_optimize = self._queries_wo_optimize
        new._n_rows_before_opt = self._n_rows_before_opt
        new._user_vec_cache_limit = self._default_user_vec_cache_limit
        new._item_vec_cache_limit = self._default_item_vec_cache_limit

        return new

    def close(self):
        """Cleanup method to delete temporary database files when they're not in use anymore."""

        if self._db_path is not None and self._db_path + self._active_table in self._shared_db_table_instances:
            self._shared_db_table_instances[self._db_path + self._active_table].remove(id(self))
            if len(self._shared_db_table_instances[self._db_path + self._active_table]) == 0:
                del self._shared_db_table_instances[self._db_path + self._active_table]
                if len(self._open_value_generators) == 0:
                    self._open_cursor().execute(f'DROP TABLE {self._active_table}')

        if self._conn is not None:
            self._conn.close()

        if self._db_path is not None and self._db_path in self._shared_db_instances:
            self._shared_db_instances[self._db_path].remove(id(self))
            if len(self._shared_db_instances[self._db_path]) == 0:
                del self._shared_db_instances[self._db_path]
                if '.tmp.' in self._db_path:
                    remove(self._db_path)
                    unregister_temp_file(self._db_path)

    def __str__(self):
        return f'[DatabaseInteractionDataset with shape ({len(self)}, {len(self.columns)})]'

    """ Auxiliary private methods"""
    def _load_from_db(self, path):
        self._log('Trying to load data from existing database...')
        self._open_connection(path, loading=True)
        self._n_rows = len(self)
        self.columns = [v[1] for v in self._conn.execute('PRAGMA table_info(interactions)').fetchall() if v[1] not in ['uid', 'iid']]

    def _load_from_csv(self, path, delimiter, has_header, encoding):

        def infer_col_types(first_line):
            projected_cols = shallowcopy(self.columns)
            # rid is never read from file - remove it to not cause issues with zip(row, projected_cols)
            projected_cols.remove('rid')

            col_types = {'rid': int}
            for value, col in zip(first_line, projected_cols):
                col_types[col] = self._infer_value_type(value)

            return col_types

        def process_segment(chunk, delimiter):
            c = self._open_cursor()
            projected_cols = shallowcopy(self.columns)
            # rid is never read from file - remove it to not cause issues with zip(row, projected_cols)
            projected_cols.remove('rid')

            try:
                csv_reader = csv.reader(chunk, delimiter=delimiter)
                records = []
                for row in csv_reader:
                    record = []
                    for value, col in zip(row, projected_cols):
                        if col is None:  # accept None cols when reading as skip cols
                            continue
                        elif col == 'interaction': # todo: do we really want this?
                            record.append(float(value))
                        else:
                            record.append(value)
                    records.append(tuple(record) + (self._n_rows,))
                    self._n_rows += 1
            except ValueError as e:
                raise Exception('Failed to process dataset chunk: Make sure that the data you\'re loading does not '
                                'have a header - if it does, set the argument "has_header" to true.\n'
                                'More details: {e}'.format(e=str(e)))

            projected_cols.append('rid')  # since rid is automatically added (explicitly) add to projected_cols
            projected_cols = [col for col in projected_cols if col is not None]  # remove any skip cols

            cols = ','.join(projected_cols)
            vars = ','.join('?' * len(projected_cols))
            c.executemany(f'INSERT INTO interactions ({cols}) VALUES ({vars})', records)
            c.close()

        self._log('Trying to load data from file, using auxiliary database...')

        # select delimiter
        orig_delimiter = delimiter
        new_delimiter = orig_delimiter
        replace_delimiter = False
        if len(orig_delimiter) > 1:  # due to csv not accept delimiters with more than 1 char
            replace_delimiter = True
            new_delimiter = ','

        # read num of lines + detect column types
        num_lines = 0
        first_line = None
        with open(path, 'r') as f:
            for row in f:
                if first_line is None and ((num_lines == 0 and has_header is False) or num_lines > 0):
                    if replace_delimiter: row = row.replace(orig_delimiter, new_delimiter)
                    first_line = next(csv.reader([row], delimiter=new_delimiter))
                num_lines += 1

        # open db connection and create table
        self._col_types = infer_col_types(first_line)
        self._open_connection(join(data_path(), f'interactions_{id(self)}.{str(random.random()).split(".")[1]}.tmp.sqlite'))
        self._build_table(self._open_cursor())
        self._n_rows = 0

        # import data into db
        chunk = []
        curr_line = 0
        with open(path, 'r', encoding=encoding) as f:
            if has_header is not False: f.readline()
            _iter = f
            if self.verbose: _iter = tqdm(f, total=num_lines, desc='Importing dataset')
            for row in _iter:
                curr_line += 1
                if replace_delimiter: row = row.replace(orig_delimiter, new_delimiter)
                chunk.append(row)
                if len(chunk) == self.chunk_size:
                    process_segment(chunk, new_delimiter)
                    chunk = []
            if len(chunk) > 0:
                process_segment(chunk, new_delimiter)

        self._build_indexes(self._open_cursor())
        self._conn.commit()

        self.columns = [col for col in self.columns if col is not None]  # remove any skip cols

    def _infer_value_type(self, value):
        try:
            if '.' in value:
                raise Exception
            int(value)
            return int
        except:
            pass

        try:
            float(value)
            return float
        except:
            pass

        return str

    def _parse_query(self, query):
        query = str(query)
        query_segments = query.split(',')
        query_conditions = []

        for segment in query_segments:
            segment = segment.strip()
            column, operator, value = segment.split(" ")
            assert column is not None or operator is not None or value is not None, \
                'Invalid get query, must be in form: column1 operator1 value1, column2 operator2 value2, ...'
            assert column in self.columns, 'Unexpected column "{c}".'.format(c=column)

            if operator not in ['>', '>=', '<=', '<', '==', '!=']:
                raise Exception(f'Unexpected operator "{operator}".')

            value = eval(value)

            try:
                col_type = self._col_types[column]
                if col_type is int:
                    if type(value) != float and type(value) != int: raise ValueError
                elif col_type is float:
                    value = float(value)
                elif col_type is list:
                    value = list(value).__repr__()
                else:
                    value = str('"' + value + '"')
            except ValueError:
                raise Exception(
                    f'Query "{query}" was failed to be parsed: check if no invalid comparisons are being made '
                    f'(column of type int being compared to a str, or vice versa).')

            if operator == '==':
                query_conditions.append(f'{column} = {value}')
            elif operator == '!=':
                query_conditions.append(f'{column} <> {value}')
            elif operator in ['>', '>=', '<=', '<']:
                query_conditions.append(f'{column} {operator} {value}')

        return query_conditions

    def _build_new_state(self, query_conditions=None, limit=None):
        if self._state_query == '':
            new_state = f'SELECT {",".join(self.columns)} FROM {self._active_table}'
            if query_conditions is not None:
                new_state += f' WHERE {" AND ".join(query_conditions)}'
        else:
            new_state = f'{self._state_query}'
            if query_conditions is not None:
                new_state += f' AND {" AND ".join(query_conditions)}'

        if limit is not None and type(limit) is int:
            new_state += f' LIMIT {limit}'

        return new_state

    def _curr_state_source(self):
        """Auxiliary method that returns self.state_query when a filter has already been applied, or the
        active table name otherwise."""
        return self._state_query if self._state_query != '' else self._active_table

    def _record_interaction(self):
        """Records a direct interaction with the current DatabaseInteractionDataset instance. If the number of direct
        interactions exceeds self._opt2_n_direct_interactions_threshold, optimize the states if possible."""
        self._n_direct_interactions += 1
        if self._n_direct_interactions >= self._opt2_n_direct_interactions_threshold \
                and len(self._open_value_generators) == 0:
            self._optimize_states()

    def _update_state(self, new_state):
        self._queries_wo_optimize += 1

        self._n_rows = self._open_cursor().execute(f'SELECT COUNT(*) FROM ({new_state})').fetchone()[0]

        self._state_query = new_state

        if self._n_rows_before_opt > 0 and self._n_rows > 0 \
                and self._queries_wo_optimize >= self._opt_n_query_threshold \
                and self._n_rows / self._n_rows_before_opt < self._opt_ratio_threshold \
                and self._n_rows_before_opt - self._n_rows >= self._opt_n_rows_threshold \
                and len(self._open_value_generators) == 0:
            self._n_rows_before_opt = self._n_rows
            self._optimize_states()

    def _optimize_states(self, skip_empty_query=True):
        """Auxiliary method that should be called inside .values() and .unique() that computes
        the results from executing the self.state_query on the current table and:
            - If this is the only instance using the current active table:
                1. Removes records from the table that are not selected by the current state query;
            - If not:
                1. Creates a reduced table;
                2. Inserts state query result into the new reduced table.
        """
        self._queries_wo_optimize = 0
        self._n_direct_interactions = 0

        if self._state_query == '' and skip_empty_query: return  # nothing to optimize
        if len(self._open_value_generators) > 0: return  # can't optimize due to value generators being read

        self._log('Optimizing state...')

        c = self._open_cursor()

        if len(self._shared_db_table_instances[self._db_path + self._active_table]) == 1:
            # this is the only instance using the current active table on this database
            if self._state_query != '':
                c.execute(f'DELETE FROM {self._active_table} '
                          f'WHERE rid NOT IN (SELECT rid FROM ({self._curr_state_source()}))')
        else:
            # this are more instances using the current active table on this database
            # optimize query path by creating a new compressed query node
            red_table = f'interactions_{str(random.random()).split(".")[1]}'

            self._build_table(c, table_name=red_table)

            col_names = ','.join(self.columns)
            c.execute(f'INSERT INTO {red_table} ({col_names}) SELECT {col_names} FROM ({self._curr_state_source()})')

            self._build_indexes(c, table_name=red_table)
            # do not delete set because there are more instances using the db, table pair
            self._shared_db_table_instances[self._db_path + self._active_table].remove(id(self))
            self._shared_db_table_instances[self._db_path + red_table] = set()
            self._shared_db_table_instances[self._db_path + red_table].add(id(self))
            self._active_table = red_table

        self._state_query = ''
        self._conn.commit()

    def _from_row_to_record(self, row, projected_columns, to_list=False):
        interaction = {k: v for k, v in zip(self.columns, row) if k in projected_columns}
        if to_list:
            list_interaction = [interaction[col] for col in projected_columns]
            if len(list_interaction) == 1: return list_interaction[0]
            else: return list_interaction
        return interaction

    def _open_connection(self, db_path, loading=False):
        """Creates a new db file and opens a new connection to it (if loading is False).
        Otherwise, if just opens a new connection to the db located at db_path."""
        if getattr(self, 'conn', None) is not None:
            self._conn.close()
        if not isfile(db_path):
            if loading:
                raise FileNotFoundError(f'No database file found at "{db_path}".')
            register_temp_file(db_path)
        self._db_path = db_path
        self._conn = sql.connect(db_path, check_same_thread=False)
        self._set_pragmas(self._conn)

    def _open_cursor(self):
        try:
            return self._conn.cursor()
        except:
            self._open_connection(self._db_path)
            return self._conn.cursor()

    def _copy_to_new_db(self, c, keep_rids=False):
        """Copies the records returned from the current state to the interactions table of the provided connection."""
        records = []
        ctd = 0
        projected_cols = shallowcopy(self.columns)
        # make sure 'rid' is the last column
        projected_cols.remove('rid')
        projected_cols.append('rid')

        cols = ','.join(projected_cols)
        vars = ','.join('?' * len(projected_cols))
        insert_query = f'INSERT INTO interactions ({cols}) VALUES ({vars})'
        for row in self.values(columns=projected_cols, to_list=True):
            if keep_rids: records.append(tuple(row))
            else: records.append(tuple(row[:-1]) + (ctd,))

            ctd += 1
            if ctd % self.chunk_size == 0:
                c.executemany(insert_query, records)
                records = []

        if len(records) > 0: c.executemany(insert_query, records)

    def _build_table(self, c, table_name='interactions'):
        py_sql_type = {
            int: 'INTEGER',
            float: 'REAL',
            str: 'TEXT',
            list: 'TEXT'
        }

        required_cols = ['rid', 'user', 'item', 'interaction']

        for req_col in required_cols:
            assert req_col in self.columns, f'Missing "{req_col}" column.'

        create_sql_statement = f'CREATE TABLE {table_name} (' \
            'rid INTEGER PRIMARY KEY,' \
            'uid INTEGER,' \
            'iid INTEGER'

        for col in self.columns:
            if col not in ['rid', 'uid', 'iid']:
                create_sql_statement += f',{col} {py_sql_type[self._col_types[col]]}'

        create_sql_statement += ')'

        c.execute(create_sql_statement)

    def _build_indexes(self, c, table_name='interactions'):
        c.execute(f'CREATE INDEX rid_idx_{table_name} ON {table_name} (rid)')

        if 'user' in self.columns:
            c.execute(f'CREATE INDEX user_idx_{table_name} ON {table_name} (user)')

        if 'item' in self.columns:
            c.execute(f'CREATE INDEX item_idx_{table_name} ON {table_name} (item)')

        self._create_internal_id_indexes(c, table_name)

    def _create_internal_id_indexes(self, c, table_name='interactions'):
        if 'user' in self.columns:
            c.execute(f'CREATE INDEX uid_idx_{table_name} ON {table_name} (uid)')

        if 'item' in self.columns:
            c.execute(f'CREATE INDEX iid_idx_{table_name} ON {table_name} (iid)')

    def _drop_internal_id_indexes(self, c, table_name='interactions'):
        if 'user' in self.columns:
            c.execute(f'DROP INDEX uid_idx_{table_name}')

        if 'item' in self.columns:
            c.execute(f'DROP INDEX iid_idx_{table_name}')

    @staticmethod
    def _set_pragmas(conn):
        conn.execute('PRAGMA journal_mode=OFF')  # no need for rollback journals - https://www.sqlite.org/pragma.html#pragma_journal_mode
        conn.commit()
        conn.execute('PRAGMA synchronous=OFF')  # no need to sync - https://www.sqlite.org/pragma.html#pragma_synchronous
        conn.commit()
        conn.execute('PRAGMA cache_size=-4000')  # increase cache size to 4000kibs - https://www.sqlite.org/pragma.html#pragma_cache_size
        conn.commit()
