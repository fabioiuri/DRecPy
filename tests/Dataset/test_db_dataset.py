from DRecPy.Dataset import InteractionDataset
from DRecPy.Dataset.db_dataset import DatabaseInteractionDataset
import pytest
import numpy as np
import os

IN_MEMORY = False

DatabaseInteractionDataset._opt_ratio_threshold = 0.8
DatabaseInteractionDataset._opt_n_query_threshold = 2
DatabaseInteractionDataset._opt_n_rows_threshold = 1
DatabaseInteractionDataset._opt2_n_direct_interactions_threshold = 3


@pytest.fixture
def resources_path():
    return os.path.join(os.path.dirname(__file__), 'resources')


@pytest.fixture
def db_interactions(resources_path):
    DatabaseInteractionDataset._shared_db_instances = {}
    DatabaseInteractionDataset._shared_db_table_instances = {}
    return InteractionDataset(os.path.join(resources_path, 'test.csv'), columns=['user', 'item', 'interaction'], in_memory=IN_MEMORY, has_header=True)


@pytest.fixture
def db_interactions_floats(resources_path):
    DatabaseInteractionDataset._shared_db_instances = {}
    DatabaseInteractionDataset._shared_db_table_instances = {}
    return InteractionDataset(os.path.join(resources_path, 'test_floats.csv'), columns=['user', 'item', 'interaction'], in_memory=IN_MEMORY, has_header=True)


@pytest.fixture
def db_interactions_int_ids(resources_path):
    DatabaseInteractionDataset._shared_db_instances = {}
    DatabaseInteractionDataset._shared_db_table_instances = {}
    return InteractionDataset(os.path.join(resources_path, 'test_int_ids.csv'), columns=['user', 'item', 'interaction'], in_memory=IN_MEMORY, has_header=True)


@pytest.fixture
def db_interactions_with_iids(resources_path):
    DatabaseInteractionDataset._shared_db_instances = {}
    DatabaseInteractionDataset._shared_db_table_instances = {}
    ds = InteractionDataset(os.path.join(resources_path, 'test.csv'), columns=['user', 'item', 'interaction'], in_memory=IN_MEMORY, has_header=True)
    ds.assign_internal_ids()
    return ds


@pytest.fixture
def db_interactions_with_mult_cols(resources_path):
    DatabaseInteractionDataset._shared_db_instances = {}
    DatabaseInteractionDataset._shared_db_table_instances = {}
    return InteractionDataset(os.path.join(resources_path, 'test_with_mult_cols.csv'), columns=['user', 'item', 'interaction', 'timestamp', 'session', 'tags'], in_memory=IN_MEMORY, has_header=True)


def check_list_equal(l1, l2):
    try:
        return len(l1) == len(l2) and sorted(l1) == sorted(l2)
    except:
        key = list(l1[0].keys())[0]
        return len(l1) == len(l2) and sorted(l1, key=lambda x: x[key]) == sorted(l2, key=lambda x: x[key])


""" Public method tests """
def test_in_memory_attr_0(db_interactions):
    assert db_interactions.in_memory == IN_MEMORY


""" __len__ """
def test_len_0(db_interactions):
    assert len(db_interactions) == 4


def test_len_1(db_interactions):
    assert len(db_interactions.select('interaction > 2')) == 3


def test_len_2(db_interactions):
    assert len(db_interactions.select('interaction > 10')) == 0


""" __str__ """
def test_str_0(db_interactions):
    assert str(db_interactions) == '[DatabaseInteractionDataset with shape (4, 4)]'


def test_str_1(db_interactions):
    assert str(db_interactions.select('rid > 2')) == '[DatabaseInteractionDataset with shape (1, 4)]'


def test_str_2(db_interactions):
    assert str(db_interactions.unique(['user'])) == '[DatabaseInteractionDataset with shape (3, 2)]'


def test_str_3(db_interactions_with_iids):
    assert str(db_interactions_with_iids) == '[DatabaseInteractionDataset with shape (4, 6)]'


def test_str_4(db_interactions_with_mult_cols):
    assert str(db_interactions_with_mult_cols) == '[DatabaseInteractionDataset with shape (4, 7)]'


def test_str_5(db_interactions_floats):
    assert str(db_interactions_floats) == '[DatabaseInteractionDataset with shape (4, 4)]'


def test_str_6(db_interactions_int_ids):
    assert str(db_interactions_int_ids) == '[DatabaseInteractionDataset with shape (4, 4)]'


""" copy """
def test_copy_0(db_interactions):
    assert id(db_interactions) != id(db_interactions.copy())


def test_copy_1(db_interactions):
    new = db_interactions.copy().select('rid > 1', copy=False)
    assert db_interactions.values_list() != new.values_list()


def test_copy_2(db_interactions):
    new = db_interactions.copy()
    assert db_interactions.values_list() == new.values_list()


""" values """
def test_values_0(db_interactions):
    assert check_list_equal([record for record in db_interactions.values()], [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])


def test_values_1(db_interactions):
    assert check_list_equal([record for record in db_interactions.values(to_list=True)], [
        ['jack', 'ps4', 3.0, 0],
        ['john', 'hard-drive', 4.0, 1],
        ['alfred', 'pen', 1.0, 2],
        ['jack', 'xbox', 5.0, 3],
    ])


def test_values_2(db_interactions):
    assert check_list_equal([record for record in db_interactions.values(columns=['item', 'user'])], [
        {'item': 'ps4', 'user': 'jack'},
        {'item': 'hard-drive', 'user': 'john'},
        {'item': 'pen', 'user': 'alfred'},
        {'item': 'xbox', 'user': 'jack'}
    ])


def test_values_3(db_interactions):
    assert check_list_equal([record for record in db_interactions.values(columns=['item', 'user'], to_list=True)], [
        ['ps4', 'jack'],
        ['hard-drive', 'john'],
        ['pen', 'alfred'],
        ['xbox', 'jack'],
    ])


def test_values_4(db_interactions):
    assert check_list_equal([record for record in db_interactions.values(columns=['user', 'item'], to_list=True)], [
        ['jack', 'ps4'],
        ['john', 'hard-drive'],
        ['alfred', 'pen'],
        ['jack', 'xbox'],
    ])


def test_values_5(db_interactions):
    try:
        next(db_interactions.values(columns=['item', 'user', 'timestamp']))
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "timestamp".'


def test_values_6(db_interactions_with_iids):
    assert check_list_equal([record for record in db_interactions_with_iids.values()], [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'uid': 0, 'iid': 0},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'uid': 2, 'iid': 2},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'uid': 0, 'iid': 3}
    ])


def test_values_7(db_interactions_with_iids):
    assert check_list_equal([record for record in db_interactions_with_iids.values(to_list=True)], [
        ['jack', 'ps4', 3.0, 0, 0, 0],
        ['john', 'hard-drive', 4.0, 1, 1, 1],
        ['alfred', 'pen', 1.0, 2, 2, 2],
        ['jack', 'xbox', 5.0, 3, 0, 3],
    ])


def test_values_8(db_interactions_floats):
    assert check_list_equal([record for record in db_interactions_floats.values()], [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.2, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.1, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.5, 'rid': 3, 'user': 'jack'}
    ])


def test_values_9(db_interactions_floats):
    assert check_list_equal([record for record in db_interactions_floats.values(to_list=True)], [
        ['jack', 'ps4', 3.0, 0],
        ['john', 'hard-drive', 4.2, 1],
        ['alfred', 'pen', 1.1, 2],
        ['jack', 'xbox', 5.5, 3],
    ])


def test_values_10(db_interactions_int_ids):
    assert check_list_equal([record for record in db_interactions_int_ids.values()], [
        {'item': 1, 'interaction': 3.0, 'rid': 0, 'user': 1},
        {'item': 2, 'interaction': 4.0, 'rid': 1, 'user': 2},
        {'item': 3, 'interaction': 1.0, 'rid': 2, 'user': 3},
        {'item': 4, 'interaction': 5.0, 'rid': 3, 'user': 1}
    ])


def test_values_11(db_interactions_int_ids):
    assert check_list_equal([record for record in db_interactions_int_ids.values(to_list=True)], [
        [1, 1, 3.0, 0],
        [2, 2, 4.0, 1],
        [3, 3, 1.0, 2],
        [1, 4, 5.0, 3],
    ])


def test_values_12(db_interactions_int_ids):
    assert check_list_equal([record for record in db_interactions_int_ids.values('interaction', to_list=True)],
                            [3.0, 4.0, 1.0, 5.0])


def test_values_13(db_interactions_int_ids):
    assert check_list_equal([record for record in db_interactions_int_ids.select('user == 2').values(to_list=True)],
                            [[2, 2, 4.0, 1]])

""" values_list """
def test_values_list_0(db_interactions):
    assert check_list_equal(db_interactions.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])


def test_values_list_1(db_interactions):
    assert check_list_equal(db_interactions.values_list(to_list=True), [
        ['jack', 'ps4', 3.0, 0],
        ['john', 'hard-drive', 4.0, 1],
        ['alfred', 'pen', 1.0, 2],
        ['jack', 'xbox', 5.0, 3],
    ])


def test_values_list_2(db_interactions):
    assert check_list_equal(db_interactions.values_list(columns=['item', 'user']), [
        {'item': 'ps4', 'user': 'jack'},
        {'item': 'hard-drive', 'user': 'john'},
        {'item': 'pen', 'user': 'alfred'},
        {'item': 'xbox', 'user': 'jack'}
    ])


def test_values_list_3(db_interactions):
    assert check_list_equal(db_interactions.values_list(columns=['item', 'user'], to_list=True), [
        ['ps4', 'jack'],
        ['hard-drive', 'john'],
        ['pen', 'alfred'],
        ['xbox', 'jack'],
    ])


def test_values_list_4(db_interactions):
    assert check_list_equal(db_interactions.values_list(columns=['user', 'item'], to_list=True), [
        ['jack', 'ps4'],
        ['john', 'hard-drive'],
        ['alfred', 'pen'],
        ['jack', 'xbox'],
    ])


def test_values_list_5(db_interactions):
    try:
        db_interactions.values_list(columns=['item', 'user', 'timestamp'])
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "timestamp".'


def test_values_list_6(db_interactions):
    assert db_interactions.select('interaction == 5').values_list() == [
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ]


def test_values_list_7(db_interactions_with_iids):
    assert check_list_equal(db_interactions_with_iids.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'uid': 0, 'iid': 0},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'uid': 2, 'iid': 2},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'uid': 0, 'iid': 3}
    ])


def test_values_list_8(db_interactions_with_iids):
    assert check_list_equal(db_interactions_with_iids.values_list(to_list=True), [
        ['jack', 'ps4', 3.0, 0, 0, 0],
        ['john', 'hard-drive', 4.0, 1, 1, 1],
        ['alfred', 'pen', 1.0, 2, 2, 2],
        ['jack', 'xbox', 5.0, 3, 0, 3],
    ])


def test_values_list_9(db_interactions_with_mult_cols):
    assert check_list_equal(db_interactions_with_mult_cols.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'timestamp': 1000.0, 'session': 5, 'tags': 'tag1;tag2'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'timestamp': 940.33, 'session': 3, 'tags': 'tag5'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'timestamp': 900.0, 'session': 2, 'tags': ''},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'timestamp': 950.52, 'session': 5, 'tags': 'tag3'}
    ])


def test_values_list_10(db_interactions_floats):
    assert check_list_equal(db_interactions_floats.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.2, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.1, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.5, 'rid': 3, 'user': 'jack'}
    ])


def test_values_list_11(db_interactions_int_ids):
    assert check_list_equal(db_interactions_int_ids.values_list(), [
        {'item': 1, 'interaction': 3.0, 'rid': 0, 'user': 1},
        {'item': 2, 'interaction': 4.0, 'rid': 1, 'user': 2},
        {'item': 3, 'interaction': 1.0, 'rid': 2, 'user': 3},
        {'item': 4, 'interaction': 5.0, 'rid': 3, 'user': 1}
    ])


""" select """
def test_select_0(db_interactions):
    new = db_interactions.select('interaction > 1')
    assert check_list_equal(db_interactions.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])
    assert check_list_equal(new.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])
    assert id(new) != id(db_interactions)


def test_select_1(db_interactions):
    same = db_interactions.select('interaction > 1', copy=False)
    assert check_list_equal(same.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])
    assert id(same) == id(db_interactions)


def test_select_2(db_interactions):
    new = db_interactions.select('interaction > 1')
    new2 = new.select('interaction < 5')
    assert check_list_equal(db_interactions.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])
    assert check_list_equal(new.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])
    assert check_list_equal(new2.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'}
    ])
    assert id(new) != id(new2) and id(new2) != id(db_interactions) and id(new) != id(db_interactions)


def test_select_3(db_interactions):
    same = db_interactions.select('interaction > 1', copy=False)
    same2 = same.select('interaction < 5', copy=False)
    assert check_list_equal(same2.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'}
    ])
    assert id(same) == id(same2) and id(same2) == id(db_interactions)


def test_select_4(db_interactions):
    new = db_interactions.select('interaction > 1, interaction < 5')
    assert check_list_equal(db_interactions.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])
    assert check_list_equal(new.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'}
    ])
    assert id(new) != id(db_interactions)


def test_select_5(db_interactions):
    same = db_interactions.select('interaction > 1, interaction < 5', copy=False)
    assert check_list_equal(same.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'}
    ])
    assert id(same) == id(db_interactions)


def test_select_6(db_interactions):
    same = db_interactions.select('interaction > 10', copy=False)
    assert same.values_list() == []
    assert id(same) == id(db_interactions)


def test_select_7(db_interactions):
    new = db_interactions.select('interaction > 1').select('interaction < 5').select('rid >= 1')
    assert check_list_equal(db_interactions.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])
    assert new.values_list() == [
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'}
    ]
    assert id(new) != id(db_interactions)


def test_select_8(db_interactions):
    try:
        db_interactions.select('interactions > 2')
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "interactions".'


def test_select_9(db_interactions):
    try:
        db_interactions.select('interaction >> 2')
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected operator ">>".'


def test_select_10(db_interactions_with_iids):
    new = db_interactions_with_iids.select('interaction > 1, interaction < 5')
    assert check_list_equal(db_interactions_with_iids.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'uid': 0, 'iid': 0},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'uid': 2, 'iid': 2},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'uid': 0, 'iid': 3}
    ])
    assert check_list_equal(new.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'uid': 0, 'iid': 0},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1}
    ])
    assert id(new) != id(db_interactions_with_iids)


def test_select_11(db_interactions_with_iids):
    new = db_interactions_with_iids.select('interaction > 1').select('interaction < 5').select('uid >= 1')
    assert check_list_equal(db_interactions_with_iids.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'uid': 0, 'iid': 0},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'uid': 2, 'iid': 2},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'uid': 0, 'iid': 3}
    ])
    assert new.values_list() == [
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1}
    ]
    assert id(new) != id(db_interactions_with_iids)


def test_select_12(db_interactions_with_mult_cols):
    assert check_list_equal(db_interactions_with_mult_cols.select('timestamp >= 950.52', copy=False).values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'timestamp': 1000.0, 'session': 5, 'tags': 'tag1;tag2'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'timestamp': 950.52, 'session': 5, 'tags': 'tag3'}
    ])


def test_select_13(db_interactions_with_mult_cols):
    assert db_interactions_with_mult_cols.select('timestamp >= 950.52', copy=False).select('tags == "tag3"', copy=False).values_list() == [
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'timestamp': 950.52, 'session': 5, 'tags': 'tag3'}
    ]


def test_select_14(db_interactions_with_mult_cols):
    assert check_list_equal(db_interactions_with_mult_cols.select('interaction != 4', copy=False).values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'timestamp': 1000.0, 'session': 5, 'tags': 'tag1;tag2'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'timestamp': 900.0, 'session': 2, 'tags': ''},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'timestamp': 950.52, 'session': 5, 'tags': 'tag3'}
    ])


def test_select_15(db_interactions):
    try:
        db_interactions.select('interaction > "14"')
        assert False
    except Exception as e:
        assert str(e) == 'Query "interaction > "14"" was failed to be parsed: check if no invalid comparisons are being ' \
                         'made (column of type int being compared to a str, or vice versa).'


def test_select_16(db_interactions_floats):
    same = db_interactions_floats.select('interaction > 1.1', copy=False)
    assert check_list_equal(same.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.2, 'rid': 1, 'user': 'john'},
        {'item': 'xbox', 'interaction': 5.5, 'rid': 3, 'user': 'jack'}
    ])
    assert id(same) == id(db_interactions_floats)


def test_select_17(db_interactions_floats):
    same = db_interactions_floats.select('interaction > 1', copy=False)
    assert check_list_equal(same.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.2, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.1, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.5, 'rid': 3, 'user': 'jack'}
    ])
    assert id(same) == id(db_interactions_floats)


def test_select_18(db_interactions_floats):
    same = db_interactions_floats.select('interaction >= 1.1', copy=False)
    assert check_list_equal(same.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.2, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.1, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.5, 'rid': 3, 'user': 'jack'}
    ])
    assert id(same) == id(db_interactions_floats)


def test_select_19(db_interactions_int_ids):
    same = db_interactions_int_ids.select('user == 1', copy=False)
    assert check_list_equal(same.values_list(), [
        {'item': 1, 'interaction': 3.0, 'rid': 0, 'user': 1},
        {'item': 4, 'interaction': 5.0, 'rid': 3, 'user': 1},
    ])
    assert id(same) == id(db_interactions_int_ids)


def test_select_20(db_interactions_int_ids):
    try:
        db_interactions_int_ids.select('user == "1"')
        assert False
    except Exception as e:
        assert str(e) == 'Query "user == "1"" was failed to be parsed: check if no invalid comparisons are being made (column of type int being compared to a str, or vice versa).'


def test_select_21(db_interactions_int_ids):
    assert check_list_equal(db_interactions_int_ids.select('user == 0').values_list(), [])


""" select_random_generator """
def test_select_random_generator_0(db_interactions):
    try:
        next(db_interactions.select_random_generator('rid > 1', seed=23))
        assert False
    except Exception as e:
        assert str(e) == 'No internal ids assigned yet.'


def test_select_random_generator_1(db_interactions_with_iids):
    assert next(db_interactions_with_iids.select_random_generator('rid > 1', seed=23))['rid'] == 3


def test_select_random_generator_2(db_interactions_with_iids):
    assert next(db_interactions_with_iids.select_random_generator('interaction > 3.0', seed=23))['interaction'] == 4.0


def test_select_random_generator_3(db_interactions_with_iids):
    try:
        next(db_interactions_with_iids.select_random_generator('interaction > 8.0', seed=23))
        assert False
    except Exception as e:
        assert str(e) == 'No records were found after applying the given query.'


def test_select_random_generator_4(db_interactions_with_iids):
    try:
        next(db_interactions_with_iids.select('interaction > 8.0').select_random_generator(seed=23))
        assert False
    except Exception as e:
        assert str(e) == 'No records were found.'


def test_select_random_generator_5(db_interactions_with_iids):
    gen = db_interactions_with_iids.select_random_generator('rid > 1', seed=23)
    next(gen)
    assert next(gen)['rid'] == 2


def test_select_random_generator_6(db_interactions_with_iids):
    gen = db_interactions_with_iids.select_random_generator(seed=23)
    assert next(gen)['rid'] == 1


def test_select_random_generator_7(db_interactions_floats):
    db_interactions_floats.assign_internal_ids()
    assert next(db_interactions_floats.select_random_generator('rid > 1', seed=23))['rid'] == 3


def test_select_random_generator_8(db_interactions_floats):
    db_interactions_floats.assign_internal_ids()
    assert next(db_interactions_floats.select_random_generator('interaction > 3.1', seed=23))['interaction'] == 4.2


def test_select_random_generator_9(db_interactions_floats):
    db_interactions_floats.assign_internal_ids()
    try:
        next(db_interactions_floats.select_random_generator('interaction > 5.6', seed=23))
        assert False
    except Exception as e:
        assert str(e) == 'No records were found after applying the given query.'


""" null_interaction_pair_generator """
def test_null_interaction_pair_generator_0(db_interactions):
    try:
        next(db_interactions.null_interaction_pair_generator(seed=23))
        assert False
    except Exception as e:
        assert str(e) == 'No internal ids assigned yet.'


def test_null_interaction_pair_generator_1(db_interactions_with_iids):
    assert next(db_interactions_with_iids.null_interaction_pair_generator(seed=23)) == (1, 0)


def test_null_interaction_pair_generator_2(db_interactions_with_iids):
    gen = db_interactions_with_iids.null_interaction_pair_generator(seed=23)
    next(gen)
    assert next(gen) == (0, 2)


def test_null_interaction_pair_generator_3(db_interactions_with_iids):
    gen = db_interactions_with_iids.null_interaction_pair_generator(seed=23)
    next(gen), next(gen)
    assert next(gen) == (1, 3)


def test_null_interaction_pair_generator_4(db_interactions_with_iids):
    gen = db_interactions_with_iids.null_interaction_pair_generator(seed=23)
    next(gen), next(gen), next(gen)
    assert next(gen) == (0, 1)


def test_null_interaction_pair_generator_5(db_interactions_with_iids):
    try:
        next(db_interactions_with_iids.select('interaction > 8').null_interaction_pair_generator(seed=23))
        assert False
    except Exception as e:
        assert str(e) == 'No records were found.'


""" select_one """
def test_select_one_0(db_interactions):
    assert db_interactions.select_one('interaction > 1') == {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'}


def test_select_one_1(db_interactions):
    assert db_interactions.select_one('interaction > 4') == {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}


def test_select_one_2(db_interactions):
    assert db_interactions.select_one('rid > 0, rid < 2') == {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'}


def test_select_one_3(db_interactions):
    assert db_interactions.select_one('rid >= 1, interaction == 4') == {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'}


def test_select_one_4(db_interactions):
    assert db_interactions.select_one('rid > 0, rid < 2, interaction == 5') is None


def test_select_one_5(db_interactions):
    try:
        db_interactions.select_one('interaction >> 2')
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected operator ">>".'


def test_select_one_6(db_interactions):
    try:
        db_interactions.select_one('uid > 2')
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "uid".'


def test_select_one_7(db_interactions_with_mult_cols):
    assert db_interactions_with_mult_cols.select('timestamp >= 950.52', copy=False).select_one('tags == "tag3"') == \
           {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'timestamp': 950.52, 'session': 5, 'tags': 'tag3'}


def test_select_one_8(db_interactions_with_iids):
    assert db_interactions_with_iids.select_one('interaction > 1') == {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'uid': 0, 'iid': 0}


def test_select_one_9(db_interactions_with_iids):
    assert db_interactions_with_iids.select_one('interaction > 4') == {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'uid': 0, 'iid': 3}


def test_select_one_10(db_interactions_with_iids):
    assert db_interactions_with_iids.select_one('uid > 0, uid < 2') == {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1}


def test_select_one_11(db_interactions_with_iids):
    assert db_interactions_with_iids.select_one('uid > 0, uid < 2, interaction == 4') == {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1}


def test_select_one_12(db_interactions_with_iids):
    assert db_interactions_with_iids.select_one('uid > 0, uid < 2, interaction == 5') is None


def test_select_one_13(db_interactions_floats):
    assert db_interactions_floats.select_one('interaction > 4') == {'item': 'hard-drive', 'interaction': 4.2, 'rid': 1, 'user': 'john'}


def test_select_one_14(db_interactions_floats):
    assert db_interactions_floats.select_one('interaction > 4.2') == {'item': 'xbox', 'interaction': 5.5, 'rid': 3, 'user': 'jack'}


""" exists """
def test_exists_0(db_interactions):
    assert db_interactions.exists('interaction == 3') is True


def test_exists_1(db_interactions):
    assert db_interactions.exists('interaction == 3.0') is True


def test_exists_2(db_interactions):
    assert db_interactions.exists('interaction == 3.5') is False


def test_exists_3(db_interactions):
    assert db_interactions.exists('rid > 1') is True


def test_exists_4(db_interactions):
    assert db_interactions.select('rid <= 1').exists('rid > 1') is False


def test_exists_5(db_interactions):
    assert db_interactions.select('rid <= 1').exists('rid >= 1') is True


def test_exists_6(db_interactions_with_mult_cols):
    assert db_interactions_with_mult_cols.exists('timestamp >= 950.52') is True


def test_exists_7(db_interactions_with_mult_cols):
    assert db_interactions_with_mult_cols.exists('timestamp >= 9500.52') is False


def test_exists_8(db_interactions_floats):
    assert db_interactions_floats.exists('interaction == 5.0') is False


def test_exists_9(db_interactions_floats):
    assert db_interactions_floats.exists('interaction == 5.5') is True


def test_exists_10(db_interactions_floats):
    assert db_interactions_floats.exists('interaction == 3') is True


""" unique """
def test_unique_0(db_interactions):
    assert check_list_equal(db_interactions.unique().values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])


def test_unique_1(db_interactions):
    assert check_list_equal(db_interactions.unique('interaction').values_list(), [
        {'interaction': 3.0, 'rid': 0},
        {'interaction': 4.0, 'rid': 1},
        {'interaction': 1.0, 'rid': 2},
        {'interaction': 5.0, 'rid': 3}
    ])


def test_unique_2(db_interactions):
    assert check_list_equal(db_interactions.unique(['interaction']).values_list(), [
        {'interaction': 3.0, 'rid': 0},
        {'interaction': 4.0, 'rid': 1},
        {'interaction': 1.0, 'rid': 2},
        {'interaction': 5.0, 'rid': 3}
    ])


def test_unique_3(db_interactions):
    assert check_list_equal(db_interactions.unique(['user']).values_list(), [
        {'user': 'jack', 'rid': 0},
        {'user': 'john', 'rid': 1},
        {'user': 'alfred', 'rid': 2}
    ])


def test_unique_4(db_interactions):
    assert check_list_equal(db_interactions.unique(['user', 'interaction']).values_list(), [
        {'interaction': 3.0, 'user': 'jack', 'rid': 0},
        {'interaction': 4.0, 'user': 'john', 'rid': 1},
        {'interaction': 1.0, 'user': 'alfred', 'rid': 2},
        {'interaction': 5.0, 'user': 'jack', 'rid': 3}
    ])


def test_unique_5(db_interactions):
    try:
        db_interactions.unique((['user', 'test']))
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "test".'


def test_unique_6(db_interactions_with_mult_cols):
    assert check_list_equal(db_interactions_with_mult_cols.unique(['tags']).values_list(), [
        {'tags': 'tag1;tag2', 'rid': 0},
        {'tags': 'tag5', 'rid': 1},
        {'tags': '', 'rid': 2},
        {'tags': 'tag3', 'rid': 3}
    ])


def test_unique_7(db_interactions_with_mult_cols):
    assert check_list_equal(db_interactions_with_mult_cols.unique('session').values_list(), [
        {'session': 5, 'rid': 0},
        {'session': 3, 'rid': 1},
        {'session': 2, 'rid': 2},
    ])


def test_unique_8(db_interactions_with_iids):
    assert check_list_equal(db_interactions_with_iids.unique().values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'uid': 0, 'iid': 0},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'uid': 2, 'iid': 2},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'uid': 0, 'iid': 3}
    ])


def test_unique_9(db_interactions_floats):
    assert check_list_equal(db_interactions_floats.unique('interaction').values_list(), [
        {'interaction': 3.0, 'rid': 0},
        {'interaction': 4.2, 'rid': 1},
        {'interaction': 1.1, 'rid': 2},
        {'interaction': 5.5, 'rid': 3}
    ])


def test_unique_10(db_interactions_int_ids):
    assert check_list_equal(db_interactions_int_ids.unique('user').values_list(), [
        {'user': 1, 'rid': 0},
        {'user': 2, 'rid': 1},
        {'user': 3, 'rid': 2},
    ])


def test_unique_11(db_interactions_int_ids):
    assert check_list_equal(db_interactions_int_ids.unique('item').values_list(), [
        {'item': 1, 'rid': 0},
        {'item': 2, 'rid': 1},
        {'item': 3, 'rid': 2},
        {'item': 4, 'rid': 3},
    ])


""" count_unique """
def test_count_unique_0(db_interactions):
    assert db_interactions.count_unique(['user']) == 3


def test_count_unique_1(db_interactions_with_mult_cols):
    assert db_interactions_with_mult_cols.count_unique('session') == 3


def test_count_unique_2(db_interactions):
    try:
        db_interactions.count_unique((['user', 'test']))
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "test".'


def test_count_unique_3(db_interactions):
    assert db_interactions.count_unique(['user', 'interaction']) == 4


""" max """
def test_max_0(db_interactions):
    assert db_interactions.max('interaction') == 5.0


def test_max_1(db_interactions):
    assert db_interactions.max('user') == 'john'


def test_max_2(db_interactions_with_mult_cols):
    assert db_interactions_with_mult_cols.max('session') == 5


def test_max_3(db_interactions):
    try:
        db_interactions.max((['user']))
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column type "<class \'list\'>".'


def test_max_4(db_interactions):
    try:
        db_interactions.max('users')
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "users".'


def test_max_5(db_interactions):
    try:
        db_interactions.max()
        assert False
    except Exception as e:
        assert str(e) == 'No column was given.'


def test_max_6(db_interactions_with_iids):
    assert db_interactions_with_iids.max('uid') == 2


def test_max_7(db_interactions_floats):
    assert db_interactions_floats.max('interaction') == 5.5


def test_max_8(db_interactions):
    assert db_interactions.max('rid') == 3


""" min """
def test_min_0(db_interactions):
    assert db_interactions.min('interaction') == 1.0


def test_min_1(db_interactions):
    assert db_interactions.min('user') == 'alfred'


def test_min_2(db_interactions_with_mult_cols):
    assert db_interactions_with_mult_cols.min('session') == 2


def test_min_3(db_interactions):
    try:
        db_interactions.min((['user']))
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column type "<class \'list\'>".'


def test_min_4(db_interactions):
    try:
        db_interactions.min('users')
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "users".'


def test_min_5(db_interactions):
    try:
        db_interactions.min()
        assert False
    except Exception as e:
        assert str(e) == 'No column was given.'


def test_min_6(db_interactions_with_iids):
    assert db_interactions_with_iids.min('uid') == 0


def test_min_7(db_interactions_floats):
    assert db_interactions_floats.min('interaction') == 1.1


def test_min_8(db_interactions):
    assert db_interactions.min('rid') == 0


""" select_user_interaction_vec """
def test_select_user_interaction_vec_0(db_interactions):
    try:
        db_interactions.select_user_interaction_vec(0).toarray().ravel()
        assert False
    except Exception as e:
        assert str(e) == 'Cannot retrieve user interaction vector without assigned internal ids.'


def test_select_user_interaction_vec_1(db_interactions):
    db_interactions = db_interactions.select('interaction > 2.5', copy=False)
    db_interactions.assign_internal_ids()
    assert np.array_equal(db_interactions.select_user_interaction_vec(0).toarray().ravel(), [3.0, 0, 5.0])
    assert np.array_equal(db_interactions.select_user_interaction_vec(1).toarray().ravel(), [0, 4.0, 0])


def test_select_user_interaction_vec_2(db_interactions):
    new = db_interactions.select('interaction > 2.5').select('interaction < 5')
    new.assign_internal_ids()
    assert np.array_equal(new.select_user_interaction_vec(0).toarray().ravel(), [3.0, 0])
    assert np.array_equal(new.select_user_interaction_vec(1).toarray().ravel(), [0, 4.0])
    try:
        new.select_user_interaction_vec(2)
        assert False
    except Exception as e:
        assert str(e) == 'User internal id 2 was not found.'


def test_select_user_interaction_vec_3(db_interactions_with_iids):
    assert np.array_equal(db_interactions_with_iids.select_user_interaction_vec(0).toarray().ravel(), [3.0, 0, 0, 5.0])


def test_select_user_interaction_vec_4(db_interactions_with_iids):
    assert np.array_equal(db_interactions_with_iids.select_user_interaction_vec(1).toarray().ravel(), [0, 4.0, 0, 0])


def test_select_user_interaction_vec_5(db_interactions_with_iids):
    assert np.array_equal(db_interactions_with_iids.select_user_interaction_vec(2).toarray().ravel(), [0, 0, 1.0, 0])


def test_select_user_interaction_vec_6(db_interactions_with_iids):
    try:
        db_interactions_with_iids.select_user_interaction_vec(3)
        assert False
    except Exception as e:
        assert str(e) == 'User internal id 3 was not found.'


def test_select_user_interaction_vec_7(db_interactions_floats):
    db_interactions_floats.assign_internal_ids()
    assert np.array_equal(db_interactions_floats.select_user_interaction_vec(0).toarray().ravel(), [3.0, 0, 0, 5.5])


def test_select_user_interaction_vec_8(db_interactions_floats):
    db_interactions_floats.assign_internal_ids()
    assert np.array_equal(db_interactions_floats.select_user_interaction_vec(1).toarray().ravel(), [0, 4.2, 0, 0])


def test_select_user_interaction_vec_9(db_interactions_floats):
    db_interactions_floats.assign_internal_ids()
    assert np.array_equal(db_interactions_floats.select_user_interaction_vec(2).toarray().ravel(), [0, 0, 1.1, 0])


""" select_item_interaction_vec """
def test_select_item_interaction_vec_0(db_interactions):
    try:
        db_interactions.select_item_interaction_vec(0).toarray().ravel()
        assert False
    except Exception as e:
        assert str(e) == 'Cannot retrieve user interaction vector without assigned internal ids.'


def test_select_item_interaction_vec_1(db_interactions):
    db_interactions = db_interactions.select('interaction > 2.5', copy=False)
    db_interactions.assign_internal_ids()
    assert np.array_equal(db_interactions.select_item_interaction_vec(0).toarray().ravel(), [3.0, 0])
    assert np.array_equal(db_interactions.select_item_interaction_vec(1).toarray().ravel(), [0, 4.0])


def test_select_item_interaction_vec_2(db_interactions):
    new = db_interactions.select('interaction > 2.5').select('interaction < 4')
    new.assign_internal_ids()
    assert np.array_equal(new.select_item_interaction_vec(0).toarray().ravel(), [3.0])
    try:
        new.select_item_interaction_vec(1)
        assert False
    except Exception as e:
        assert str(e) == 'Item internal id 1 was not found.'


def test_select_item_interaction_vec_3(db_interactions_with_iids):
    assert np.array_equal(db_interactions_with_iids.select_item_interaction_vec(0).toarray().ravel(), [3.0, 0, 0])


def test_select_item_interaction_vec_4(db_interactions_with_iids):
    assert np.array_equal(db_interactions_with_iids.select_item_interaction_vec(1).toarray().ravel(), [0, 4.0, 0])


def test_select_item_interaction_vec_5(db_interactions_with_iids):
    assert np.array_equal(db_interactions_with_iids.select_item_interaction_vec(2).toarray().ravel(), [0, 0, 1.0])


def test_select_item_interaction_vec_6(db_interactions_with_iids):
    assert np.array_equal(db_interactions_with_iids.select_item_interaction_vec(3).toarray().ravel(), [5.0, 0, 0])


def test_select_item_interaction_vec_7(db_interactions_with_iids):
    try:
        db_interactions_with_iids.select_item_interaction_vec(4)
        assert False
    except Exception as e:
        assert str(e) == 'Item internal id 4 was not found.'


def test_select_item_interaction_vec_8(db_interactions_floats):
    db_interactions_floats.assign_internal_ids()
    assert np.array_equal(db_interactions_floats.select_item_interaction_vec(0).toarray().ravel(), [3.0, 0, 0])


def test_select_item_interaction_vec_9(db_interactions_floats):
    db_interactions_floats.assign_internal_ids()
    assert np.array_equal(db_interactions_floats.select_item_interaction_vec(1).toarray().ravel(), [0, 4.2, 0])


def test_select_item_interaction_vec_10(db_interactions_floats):
    db_interactions_floats.assign_internal_ids()
    assert np.array_equal(db_interactions_floats.select_item_interaction_vec(2).toarray().ravel(), [0, 0, 1.1])


def test_select_item_interaction_vec_11(db_interactions_floats):
    db_interactions_floats.assign_internal_ids()
    assert np.array_equal(db_interactions_floats.select_item_interaction_vec(3).toarray().ravel(), [5.5, 0, 0])


""" user_to_uid """
def test_user_to_uid_0(db_interactions_with_iids):
    assert db_interactions_with_iids.user_to_uid('jack') == 0
    assert db_interactions_with_iids.user_to_uid('john') == 1
    assert db_interactions_with_iids.user_to_uid('alfred') == 2


def test_user_to_uid_1(db_interactions_with_iids):
    assert db_interactions_with_iids.user_to_uid('bla') is None


def test_user_to_uid_2(db_interactions_with_iids):
    assert db_interactions_with_iids.select('user == "alfred"').user_to_uid('alfred') == 2


def test_user_to_uid_3(db_interactions_with_iids):
    assert db_interactions_with_iids.select('user == "alfred"').user_to_uid('jack') == 0


def test_user_to_uid_4(db_interactions_with_iids):
    assert db_interactions_with_iids.select('rid > 2').user_to_uid('jack') == 0


def test_user_to_uid_5(db_interactions):
    try:
        db_interactions.user_to_uid('jack')
        assert False
    except Exception as e:
        assert str(e) == 'No internal ids assigned yet.'


def test_user_to_uid_6(db_interactions_int_ids):
    db_interactions_int_ids.assign_internal_ids()
    assert db_interactions_int_ids.user_to_uid(1) == 0
    assert db_interactions_int_ids.user_to_uid(2) == 1
    assert db_interactions_int_ids.user_to_uid(3) == 2


def test_user_to_uid_7(db_interactions_int_ids):
    db_interactions_int_ids.assign_internal_ids()
    try:
        db_interactions_int_ids.user_to_uid('jack')
        assert False
        assert False
    except Exception as e:
        assert str(e) == "The provided user type does not match the inferred type (expected: int, found: <class 'str'>"


""" uid_to_user """
def test_uid_to_user_0(db_interactions_with_iids):
    assert db_interactions_with_iids.uid_to_user(0) == 'jack'
    assert db_interactions_with_iids.uid_to_user(1) == 'john'
    assert db_interactions_with_iids.uid_to_user(2) == 'alfred'


def test_uid_to_user_1(db_interactions_with_iids):
    assert db_interactions_with_iids.uid_to_user(3) is None


def test_uid_to_user_2(db_interactions_with_iids):
    assert db_interactions_with_iids.select('user == "alfred"').uid_to_user(0) == 'jack'


def test_uid_to_user_3(db_interactions_with_iids):
    assert db_interactions_with_iids.select('user == "alfred"').uid_to_user(1) == 'john'


def test_uid_to_user_4(db_interactions_with_iids):
    assert db_interactions_with_iids.select('rid > 2').uid_to_user(1) == 'john'


def test_uid_to_user_5(db_interactions):
    try:
        db_interactions.uid_to_user(0)
        assert False
    except Exception as e:
        assert str(e) == 'No internal ids assigned yet.'


def test_uid_to_user_6(db_interactions_int_ids):
    db_interactions_int_ids.assign_internal_ids()
    assert db_interactions_int_ids.uid_to_user(0) == 1
    assert db_interactions_int_ids.uid_to_user(1) == 2
    assert db_interactions_int_ids.uid_to_user(2) == 3


""" item_to_iid """
def test_item_to_iid_0(db_interactions_with_iids):
    assert db_interactions_with_iids.item_to_iid('ps4') == 0
    assert db_interactions_with_iids.item_to_iid('hard-drive') == 1
    assert db_interactions_with_iids.item_to_iid('pen') == 2
    assert db_interactions_with_iids.item_to_iid('xbox') == 3


def test_item_to_iid_1(db_interactions_with_iids):
    assert db_interactions_with_iids.item_to_iid('bla') is None


def test_item_to_iid_2(db_interactions_with_iids):
    assert db_interactions_with_iids.select('item == "hard-drive"').item_to_iid('hard-drive') == 1


def test_item_to_iid_3(db_interactions_with_iids):
    assert db_interactions_with_iids.select('item == "hard-drive"').item_to_iid('pen') == 2


def test_item_to_iid_4(db_interactions_with_iids):
    assert db_interactions_with_iids.select('rid > 2').item_to_iid('xbox') == 3


def test_item_to_iid_5(db_interactions):
    try:
        db_interactions.item_to_iid('ps4')
        assert False
    except Exception as e:
        assert str(e) == 'No internal ids assigned yet.'


def test_item_to_iid_6(db_interactions_int_ids):
    db_interactions_int_ids.assign_internal_ids()
    assert db_interactions_int_ids.item_to_iid(1) == 0
    assert db_interactions_int_ids.item_to_iid(2) == 1
    assert db_interactions_int_ids.item_to_iid(3) == 2
    assert db_interactions_int_ids.item_to_iid(4) == 3


def test_item_to_iid_7(db_interactions_int_ids):
    db_interactions_int_ids.assign_internal_ids()
    try:
        db_interactions_int_ids.item_to_iid('ps4')
        assert False
    except Exception as e:
        assert str(e) == "The provided item type does not match the inferred type (expected: int, found: <class 'str'>"


""" iid_to_item """
def test_iid_to_item_0(db_interactions_with_iids):
    assert db_interactions_with_iids.iid_to_item(0) == 'ps4'
    assert db_interactions_with_iids.iid_to_item(1) == 'hard-drive'
    assert db_interactions_with_iids.iid_to_item(2) == 'pen'
    assert db_interactions_with_iids.iid_to_item(3) == 'xbox'


def test_iid_to_item_1(db_interactions_with_iids):
    assert db_interactions_with_iids.iid_to_item(4) is None


def test_iid_to_item_2(db_interactions_with_iids):
    assert db_interactions_with_iids.select('iid == 0').iid_to_item(0) == 'ps4'


def test_iid_to_item_3(db_interactions_with_iids):
    assert db_interactions_with_iids.select('iid == 0').iid_to_item(1) == 'hard-drive'


def test_iid_to_item_4(db_interactions_with_iids):
    assert db_interactions_with_iids.select('rid > 1').iid_to_item(3) == 'xbox'


def test_iid_to_item_5(db_interactions):
    try:
        db_interactions.iid_to_item(0)
        assert False
    except Exception as e:
        assert str(e) == 'No internal ids assigned yet.'


def test_iid_to_item_6(db_interactions_int_ids):
    db_interactions_int_ids.assign_internal_ids()
    assert db_interactions_int_ids.iid_to_item(0) == 1
    assert db_interactions_int_ids.iid_to_item(1) == 2
    assert db_interactions_int_ids.iid_to_item(2) == 3
    assert db_interactions_int_ids.iid_to_item(3) == 4


""" drop """
def test_drop_0(db_interactions):
    assert check_list_equal(db_interactions.drop([0, 2]).values_list(), [
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])


def test_drop_1(db_interactions):
    assert db_interactions.drop([0, 2]).select('interaction > 4').values_list() == [
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ]


def test_drop_2(db_interactions):
    assert db_interactions.select('interaction > 4').drop([0, 2]).values_list() == [
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ]


def test_drop_3(db_interactions):
    assert db_interactions.select('interaction > 1', copy=False).select('interaction > 2', copy=False)\
               .drop([0]).drop([3]).drop([0]).values_list() == [
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
    ]


def test_drop_4(db_interactions_with_mult_cols):
    assert db_interactions_with_mult_cols.select('interaction > 1', copy=False).select('interaction > 2', copy=False)\
               .drop([0]).drop([3]).drop([0]).values_list() == [
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'timestamp': 940.33, 'session': 3, 'tags': 'tag5'},
    ]


def test_drop_5(db_interactions):
    assert check_list_equal(db_interactions.drop([0, 2], keep=True).values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
    ])


""" apply """
def test_apply_0(db_interactions):
    db_interactions.apply('interaction', lambda x: 1 if x > 2.5 else 0)
    assert check_list_equal([record for record in db_interactions.values()], [
        {'item': 'ps4', 'interaction': 1, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 1, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 1, 'rid': 3, 'user': 'jack'}
    ])


def test_apply_1(db_interactions):
    db_interactions.apply('interaction', lambda x: x / 5)
    assert check_list_equal([record for record in db_interactions.values()], [
        {'item': 'ps4', 'interaction': 0.6, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 0.8, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 0.2, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 1, 'rid': 3, 'user': 'jack'}
    ])


def test_apply_2(db_interactions_with_mult_cols):
    db_interactions_with_mult_cols.apply('tags', hash)
    assert check_list_equal(db_interactions_with_mult_cols.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'timestamp': 1000.0, 'session': 5,  'tags': hash('tag1;tag2')},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'timestamp': 940.33, 'session': 3, 'tags': hash('tag5')},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'timestamp': 900.0, 'session': 2, 'tags': hash('')},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'timestamp': 950.52, 'session': 5, 'tags': hash('tag3')}
    ])


def test_apply_3(db_interactions):
    try:
        db_interactions.apply('bla', lambda x: x)
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "bla".'


def test_apply_4(db_interactions):
    try:
        db_interactions.apply('rid', lambda x: x)
        assert False
    except Exception as e:
        assert str(e) == 'Column "rid" is read-only.'


def test_apply_5(db_interactions):
    db_interactions.assign_internal_ids()
    try:
        db_interactions.apply('uid', lambda x: x)
        assert False
    except Exception as e:
        assert str(e) == 'Column "uid" is read-only.'


def test_apply_6(db_interactions):
    db_interactions.assign_internal_ids()
    try:
        db_interactions.apply('iid', lambda x: x)
        assert False
    except Exception as e:
        assert str(e) == 'Column "iid" is read-only.'


def test_apply_7(db_interactions):
    db_interactions.assign_internal_ids()
    try:
        db_interactions.apply('user', lambda x: x)
        assert False
    except Exception as e:
        assert str(e) == 'Column "user" is read-only.'


def test_apply_8(db_interactions):
    db_interactions.assign_internal_ids()
    try:
        db_interactions.apply('item', lambda x: x)
        assert False
    except Exception as e:
        assert str(e) == 'Column "item" is read-only.'


def test_apply_9(db_interactions):
    try:
        db_interactions.apply('interaction', lambda x: x/0)
        assert False
    except Exception as e:
        assert str(e) == 'Failed to apply operation on column "interaction". Details: division by zero'


def test_apply_10(db_interactions):
    try:
        db_interactions.apply('interaction', None)
        assert False
    except Exception as e:
        assert str(e) == 'Failed to apply operation on column "interaction". Details: \'NoneType\' object is not callable'


def test_apply_11(db_interactions_with_mult_cols):
    try:
        db_interactions_with_mult_cols.apply('tags', lambda x: x.split())
        assert False
    except Exception as e:
        assert str(e) == 'Failed to apply operation on column "tags". Details: New column type "<class \'list\'>" is not supported. Supported types: [<class \'int\'>, <class \'float\'>, <class \'str\'>].'


def test_apply_12(db_interactions):
    db_interactions_old = db_interactions.copy()
    db_interactions.apply('interaction', lambda x: 1 if x > 2.5 else 0)
    assert check_list_equal([record for record in db_interactions.values()], [
        {'item': 'ps4', 'interaction': 1, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 1, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 1, 'rid': 3, 'user': 'jack'}
    ])
    assert check_list_equal([record for record in db_interactions_old.values()], [
        {'item': 'ps4', 'interaction': 3, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5, 'rid': 3, 'user': 'jack'}
    ])


""" save """
def test_save_0(db_interactions):
    db_interactions.save('testtmp.csv', write_header=True)
    tmp_file = open('testtmp.csv', 'r')
    try:
        assert tmp_file.readlines() == [
            'user,item,interaction\n',
            'jack,ps4,3\n',
            'john,hard-drive,4\n',
            'alfred,pen,1\n',
            'jack,xbox,5\n'
        ]
    finally:
        tmp_file.close()
        os.remove('testtmp.csv')


def test_save_1(db_interactions_with_iids):
    db_interactions_with_iids.save('testtmp.csv')
    tmp_file = open('testtmp.csv', 'r')
    try:
        assert tmp_file.readlines() == [
            'jack,ps4,3\n',
            'john,hard-drive,4\n',
            'alfred,pen,1\n',
            'jack,xbox,5\n'
        ]
    finally:
        tmp_file.close()
        os.remove('testtmp.csv')


def test_save_2(db_interactions):
    db_interactions.save('testtmp.csv')
    try:
        assert check_list_equal(InteractionDataset('testtmp.csv', columns=['user', 'item', 'interaction'], in_memory=IN_MEMORY).values_list(), [
            {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
            {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
            {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
            {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
        ])
    finally:
        os.remove('testtmp.csv')


def test_save_3(db_interactions_with_iids):
    db_interactions_with_iids.save('testtmp.csv')
    try:
        assert check_list_equal(InteractionDataset('testtmp.csv', columns=['user', 'item', 'interaction'], in_memory=IN_MEMORY).values_list(), [
            {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
            {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
            {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
            {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
        ])
    finally:
        os.remove('testtmp.csv')


def test_save_4(db_interactions_with_iids):
    db_interactions_with_iids.save('testtmp.csv')
    try:
        assert check_list_equal(InteractionDataset('testtmp.csv', columns=['user', 'item', 'interaction'], in_memory=IN_MEMORY).values_list(), [
            {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
            {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
            {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
            {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
        ])
    finally:
        os.remove('testtmp.csv')


def test_save_5(db_interactions_with_iids):
    db_interactions_with_iids.select('interaction > 3').save('testtmp.csv')
    try:
        assert check_list_equal(InteractionDataset('testtmp.csv', columns=['user', 'item', 'interaction'], in_memory=IN_MEMORY).values_list(), [
            {'item': 'hard-drive', 'interaction': 4.0, 'rid': 0, 'user': 'john'},
            {'item': 'xbox', 'interaction': 5.0, 'rid': 1, 'user': 'jack'}
        ])
    finally:
        os.remove('testtmp.csv')


def test_save_6(db_interactions_with_mult_cols):
    db_interactions_with_mult_cols.save('testtmp.csv')
    try:
        assert check_list_equal(InteractionDataset('testtmp.csv', columns=['user', 'item', 'interaction', 'timestamp', 'session', 'tags'], in_memory=IN_MEMORY).values_list(), [
            {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'timestamp': 1000.0, 'session': 5, 'tags': 'tag1;tag2'},
            {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'timestamp': 940.33, 'session': 3, 'tags': 'tag5'},
            {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'timestamp': 900.0, 'session': 2, 'tags': ''},
            {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'timestamp': 950.52, 'session': 5, 'tags': 'tag3'}
        ])
    finally:
        os.remove('testtmp.csv')


def test_save_7(db_interactions_with_mult_cols):
    db_interactions_with_mult_cols.save('testtmp.csv')
    try:
        assert check_list_equal(InteractionDataset('testtmp.csv', columns=['user', 'item', 'interaction', 'timestamp', None, 'tags'], in_memory=IN_MEMORY).values_list(), [
            {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'timestamp': 1000.0, 'tags': 'tag1;tag2'},
            {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'timestamp': 940.33, 'tags': 'tag5'},
            {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'timestamp': 900.0, 'tags': ''},
            {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'timestamp': 950.52, 'tags': 'tag3'}
        ])
    finally:
        os.remove('testtmp.csv')


def test_save_8(db_interactions_floats):
    db_interactions_floats.save('testtmp.csv', write_header=True)
    tmp_file = open('testtmp.csv', 'r')
    try:
        assert tmp_file.readlines() == [
            'user,item,interaction\n',
            'jack,ps4,3.0\n',
            'john,hard-drive,4.2\n',
            'alfred,pen,1.1\n',
            'jack,xbox,5.5\n'
        ]
    finally:
        tmp_file.close()
        os.remove('testtmp.csv')


def test_save_9(db_interactions_int_ids):
    db_interactions_int_ids.save('testtmp.csv', write_header=True)
    tmp_file = open('testtmp.csv', 'r')
    try:
        assert tmp_file.readlines() == [
            'user,item,interaction\n',
            '1,1,3\n',
            '2,2,4\n',
            '3,3,1\n',
            '1,4,5\n'
        ]
    finally:
        tmp_file.close()
        os.remove('testtmp.csv')


""" assign_internal_ids """
def test_assign_internal_ids_0(db_interactions):
    assert db_interactions.has_internal_ids is False


def test_assign_internal_ids_1(db_interactions):
    db_interactions.assign_internal_ids()
    assert db_interactions.has_internal_ids is True


def test_assign_internal_ids_2(db_interactions_with_iids):
    assert db_interactions_with_iids.has_internal_ids is True


def test_assign_internal_ids_3(db_interactions):
    db_interactions.assign_internal_ids()
    assert check_list_equal(db_interactions.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'uid': 0, 'iid': 0},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'uid': 2, 'iid': 2},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'uid': 0, 'iid': 3}
    ])


""" remove_internal_ids """
def test_remove_internal_ids_0(db_interactions):
    db_interactions.assign_internal_ids()
    db_interactions.remove_internal_ids()
    assert db_interactions.has_internal_ids is False


def test_remove_internal_ids_1(db_interactions_with_iids):
    db_interactions_with_iids.remove_internal_ids()
    assert db_interactions_with_iids.has_internal_ids is False


def test_remove_internal_ids_2(db_interactions_with_iids):
    db_interactions_with_iids.remove_internal_ids()
    assert check_list_equal(db_interactions_with_iids.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])


""" Specific tests """
""" save (db) """
def test_save_db_0(db_interactions_with_mult_cols):
    db_interactions_with_mult_cols.save('testtmp.sqlite')
    db = InteractionDataset('testtmp.sqlite', in_memory=IN_MEMORY)
    try:
        assert check_list_equal(db.values_list(), [
            {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'timestamp': 1000.0, 'session': 5, 'tags': 'tag1;tag2'},
            {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'timestamp': 940.33, 'session': 3, 'tags': 'tag5'},
            {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'timestamp': 900.0, 'session': 2, 'tags': ''},
            {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'timestamp': 950.52, 'session': 5, 'tags': 'tag3'}
        ])
    finally:
        db.close()
        os.remove('testtmp.sqlite')


def test_save_db_1(db_interactions_floats):
    db_interactions_floats.save('testtmp.sqlite')
    db = InteractionDataset('testtmp.sqlite', in_memory=IN_MEMORY)
    try:
        assert check_list_equal(db.values_list(), [
            {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
            {'item': 'hard-drive', 'interaction': 4.2, 'rid': 1, 'user': 'john'},
            {'item': 'pen', 'interaction': 1.1, 'rid': 2, 'user': 'alfred'},
            {'item': 'xbox', 'interaction': 5.5, 'rid': 3, 'user': 'jack'}
        ])
    finally:
        db.close()
        os.remove('testtmp.sqlite')


def test_save_db_2(db_interactions_int_ids):
    db_interactions_int_ids.save('testtmp.sqlite')
    db = InteractionDataset('testtmp.sqlite', in_memory=IN_MEMORY)
    try:
        assert check_list_equal(db.values_list(), [
            {'item': 1, 'interaction': 3.0, 'rid': 0, 'user': 1},
            {'item': 2, 'interaction': 4.0, 'rid': 1, 'user': 2},
            {'item': 3, 'interaction': 1.0, 'rid': 2, 'user': 3},
            {'item': 4, 'interaction': 5.0, 'rid': 3, 'user': 1}
        ])
    finally:
        db.close()
        os.remove('testtmp.sqlite')


""" type """
def test_type_0(db_interactions):
    assert type(db_interactions) is DatabaseInteractionDataset


def test_type_1(db_interactions):
    assert type(db_interactions.select('interaction > 1')) is DatabaseInteractionDataset


def test_type_2(db_interactions):
    assert type(db_interactions.unique('interaction')) is DatabaseInteractionDataset


""" shared instances """
def test_shared_instances_0(db_interactions):
    assert DatabaseInteractionDataset._shared_db_instances == {db_interactions._db_path: {id(db_interactions)}}
    assert DatabaseInteractionDataset._shared_db_table_instances == {db_interactions._db_path + db_interactions._active_table: {id(db_interactions)}}


def test_shared_instances_1(db_interactions):
    other = db_interactions.select('interaction > 1')
    assert DatabaseInteractionDataset._shared_db_instances == {db_interactions._db_path: {id(db_interactions), id(other)}}
    assert DatabaseInteractionDataset._shared_db_table_instances == {db_interactions._db_path + db_interactions._active_table: {id(db_interactions), id(other)}}


def test_shared_instances_2(db_interactions):
    other = db_interactions.select('interaction > 1', copy=False)
    assert id(other) == id(db_interactions)
    assert DatabaseInteractionDataset._shared_db_instances == {db_interactions._db_path: {id(db_interactions)}}
    assert DatabaseInteractionDataset._shared_db_table_instances == {db_interactions._db_path + db_interactions._active_table: {id(db_interactions)}}


def test_shared_instances_3(db_interactions):
    other = db_interactions.select('interaction > 1')
    other2 = other.select('interaction > 2')
    other3 = other2.select('interaction > 3')
    other4 = other.select('interaction > 3')
    assert DatabaseInteractionDataset._shared_db_instances == {db_interactions._db_path: {id(db_interactions), id(other), id(other2), id(other3), id(other4)}}
    assert DatabaseInteractionDataset._shared_db_table_instances == {db_interactions._db_path + db_interactions._active_table: {id(db_interactions), id(other)},
                                                                     db_interactions._db_path + other2._active_table: {id(other2), id(other3)},
                                                                     db_interactions._db_path + other4._active_table: {id(other4)}}


""" reduced table / optimizations """
def test_reduced_table_0(db_interactions):
    assert db_interactions._active_table == 'interactions'


def test_reduced_table_1(db_interactions):
    new = db_interactions.select('interaction > 1')
    assert new._active_table == 'interactions'
    assert new._db_path == db_interactions._db_path


def test_reduced_table_2(db_interactions):
    new = db_interactions.select('interaction > 1').select('interaction > 2')
    assert new._active_table != 'interactions'
    assert 'interactions_' in new._active_table
    assert new._db_path == db_interactions._db_path


def test_reduced_table_4(db_interactions):
    same = db_interactions.select('interaction > 1', copy=False)
    assert same._active_table == 'interactions'


def test_reduced_table_5(db_interactions):
    same = db_interactions.select('interaction > 1', copy=False).select('interaction > 2', copy=False)
    assert same._active_table == 'interactions'


def test_reduced_table_6(db_interactions):
    new = db_interactions.select('interaction > 1').select('interaction > 2')
    new2 = new.select('interaction > 3')
    new3 = new.select('interaction > 3').select('interaction > 4')
    new4 = db_interactions.select('interaction > 1').select('interaction > 2')
    # reduced tables should be created for new, new2 (shared with new), new3, new4
    assert new._active_table != 'interactions' and 'interactions_' in new._active_table
    assert new2._active_table != 'interactions' and 'interactions_' in new2._active_table
    assert new3._active_table != 'interactions' and 'interactions_' in new3._active_table
    assert new4._active_table != 'interactions' and 'interactions_' in new4._active_table
    # some should share reductions whilst others should create new ones
    assert new._active_table == new2._active_table
    assert new._active_table != new3._active_table
    assert new._active_table != new4._active_table
    # all should always share the same db path (for saving db purposes)
    assert new._db_path == db_interactions._db_path
    assert new2._db_path == db_interactions._db_path
    assert new3._db_path == db_interactions._db_path
    assert new4._db_path == db_interactions._db_path
    # check n_direct_interactions
    assert db_interactions._n_direct_interactions == 2
    assert new._n_direct_interactions == 2
    assert new2._n_direct_interactions == 0
    assert new3._n_direct_interactions == 0
    assert new4._n_direct_interactions == 0


def test_reduced_table_7(db_interactions):
    new = db_interactions.select('interaction > 1')
    db_interactions.select_one('user == "haley"')
    new.select('interaction > 2')
    new.select_one('user == "joseph"')
    # check n_direct_interactions and state_query before the call that triggers the optimization
    assert new._state_query != ''
    assert new._n_direct_interactions == 2

    new.select('interaction < 4')

    # check n_direct_interactions
    assert db_interactions._n_direct_interactions == 2
    assert new._n_direct_interactions == 0
    # check state_query
    assert new._state_query == ''
