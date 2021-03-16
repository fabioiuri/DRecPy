from DRecPy.Dataset import InteractionDataset
import pytest
import numpy as np
import pandas as pd
import os

IN_MEMORY = True


@pytest.fixture
def resources_path():
    return os.path.join(os.path.dirname(__file__), 'resources')


@pytest.fixture
def mem_interactions(resources_path):
    return InteractionDataset(os.path.join(resources_path, 'test.csv'), columns=['user', 'item', 'interaction'], in_memory=IN_MEMORY, has_header=True)


@pytest.fixture
def mem_interactions_floats(resources_path):
    return InteractionDataset(os.path.join(resources_path, 'test_floats.csv'), columns=['user', 'item', 'interaction'], in_memory=IN_MEMORY, has_header=True)


@pytest.fixture
def mem_interactions_int_ids(resources_path):
    return InteractionDataset(os.path.join(resources_path, 'test_int_ids.csv'), columns=['user', 'item', 'interaction'], in_memory=IN_MEMORY, has_header=True)


@pytest.fixture
def mem_interactions_with_iids(resources_path):
    ds = InteractionDataset(os.path.join(resources_path, 'test.csv'), columns=['user', 'item', 'interaction'], in_memory=IN_MEMORY, has_header=True)
    ds.assign_internal_ids()
    return ds


@pytest.fixture
def mem_interactions_with_mult_cols(resources_path):
    return InteractionDataset(os.path.join(resources_path, 'test_with_mult_cols.csv'), columns=['user', 'item', 'interaction', 'timestamp', 'session', 'tags'], in_memory=IN_MEMORY, has_header=True)


@pytest.fixture()
def mem_df(resources_path):
    return pd.read_csv(os.path.join(resources_path, 'test.csv'))


@pytest.fixture()
def mem_df_floats(resources_path):
    return pd.read_csv(os.path.join(resources_path, 'test_floats.csv'))


@pytest.fixture()
def mem_df_int_ids(resources_path):
    return pd.read_csv(os.path.join(resources_path, 'test_int_ids.csv'))


@pytest.fixture()
def mem_df_with_columns(resources_path):
    return pd.read_csv(os.path.join(resources_path, 'test.csv'), names=['user', 'item', 'interaction'], skiprows=1)


def check_list_equal(l1, l2):
    try:
        return len(l1) == len(l2) and sorted(l1) == sorted(l2)
    except:
        key = list(l1[0].keys())[0]
        return len(l1) == len(l2) and sorted(l1, key=lambda x: x[key]) == sorted(l2, key=lambda x: x[key])


def check_dict_equal(d1, d2):
    intersect_keys = set(d1.keys()).intersection(set(d2.keys()))
    modified = {o: (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    return len(modified) == 0


""" Public method tests """
""" in_memory attr """
def test_in_memory_attr_0(mem_interactions):
    assert mem_interactions.in_memory == IN_MEMORY


""" __len__ """
def test_len_0(mem_interactions):
    assert len(mem_interactions) == 4


def test_len_1(mem_interactions):
    assert len(mem_interactions.select('interaction > 2')) == 3


def test_len_2(mem_interactions):
    assert len(mem_interactions.select('interaction > 10')) == 0


""" __str__ """
def test_str_0(mem_interactions):
    assert str(mem_interactions) == '[MemoryInteractionDataset with shape (4, 4)]'


def test_str_1(mem_interactions):
    assert str(mem_interactions.select('rid > 2')) == '[MemoryInteractionDataset with shape (1, 4)]'


def test_str_2(mem_interactions):
    assert str(mem_interactions.unique(['user'])) == '[MemoryInteractionDataset with shape (3, 2)]'


def test_str_3(mem_interactions_with_iids):
    assert str(mem_interactions_with_iids) == '[MemoryInteractionDataset with shape (4, 6)]'


def test_str_4(mem_interactions_with_mult_cols):
    assert str(mem_interactions_with_mult_cols) == '[MemoryInteractionDataset with shape (4, 7)]'


def test_str_5(mem_interactions_floats):
    assert str(mem_interactions_floats) == '[MemoryInteractionDataset with shape (4, 4)]'


def test_str_6(mem_interactions_int_ids):
    assert str(mem_interactions_int_ids) == '[MemoryInteractionDataset with shape (4, 4)]'


""" copy """
def test_copy_0(mem_interactions):
    assert id(mem_interactions) != id(mem_interactions.copy())


def test_copy_1(mem_interactions):
    new = mem_interactions.copy().select('rid > 1', copy=False)
    assert mem_interactions.values_list() != new.values_list()


def test_copy_2(mem_interactions):
    new = mem_interactions.copy()
    assert mem_interactions.values_list() == new.values_list()


""" values """
def test_values_0(mem_interactions):
    assert check_list_equal([record for record in mem_interactions.values()], [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])


def test_values_1(mem_interactions):
    assert check_list_equal([record for record in mem_interactions.values(to_list=True)], [
        ['jack', 'ps4', 3.0, 0],
        ['john', 'hard-drive', 4.0, 1],
        ['alfred', 'pen', 1.0, 2],
        ['jack', 'xbox', 5.0, 3],
    ])


def test_values_2(mem_interactions):
    assert check_list_equal([record for record in mem_interactions.values(columns=['item', 'user'])], [
        {'item': 'ps4', 'user': 'jack'},
        {'item': 'hard-drive', 'user': 'john'},
        {'item': 'pen', 'user': 'alfred'},
        {'item': 'xbox', 'user': 'jack'}
    ])


def test_values_3(mem_interactions):
    assert check_list_equal([record for record in mem_interactions.values(columns=['item', 'user'], to_list=True)], [
        ['ps4', 'jack'],
        ['hard-drive', 'john'],
        ['pen', 'alfred'],
        ['xbox', 'jack'],
    ])


def test_values_4(mem_interactions):
    assert check_list_equal([record for record in mem_interactions.values(columns=['user', 'item'], to_list=True)], [
        ['jack', 'ps4'],
        ['john', 'hard-drive'],
        ['alfred', 'pen'],
        ['jack', 'xbox'],
    ])


def test_values_5(mem_interactions):
    try:
        next(mem_interactions.values(columns=['item', 'user', 'timestamp']))
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "timestamp".'


def test_values_6(mem_interactions_with_iids):
    assert check_list_equal([record for record in mem_interactions_with_iids.values()], [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'uid': 0, 'iid': 0},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'uid': 2, 'iid': 2},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'uid': 0, 'iid': 3}
    ])


def test_values_7(mem_interactions_with_iids):
    assert check_list_equal([record for record in mem_interactions_with_iids.values(to_list=True)], [
        ['jack', 'ps4', 3.0, 0, 0, 0],
        ['john', 'hard-drive', 4.0, 1, 1, 1],
        ['alfred', 'pen', 1.0, 2, 2, 2],
        ['jack', 'xbox', 5.0, 3, 0, 3],
    ])


def test_values_8(mem_interactions_floats):
    assert check_list_equal([record for record in mem_interactions_floats.values()], [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.2, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.1, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.5, 'rid': 3, 'user': 'jack'}
    ])


def test_values_9(mem_interactions_floats):
    assert check_list_equal([record for record in mem_interactions_floats.values(to_list=True)], [
        ['jack', 'ps4', 3.0, 0],
        ['john', 'hard-drive', 4.2, 1],
        ['alfred', 'pen', 1.1, 2],
        ['jack', 'xbox', 5.5, 3],
    ])


def test_values_10(mem_interactions_int_ids):
    assert check_list_equal([record for record in mem_interactions_int_ids.values()], [
        {'item': 1, 'interaction': 3.0, 'rid': 0, 'user': 1},
        {'item': 2, 'interaction': 4.0, 'rid': 1, 'user': 2},
        {'item': 3, 'interaction': 1.0, 'rid': 2, 'user': 3},
        {'item': 4, 'interaction': 5.0, 'rid': 3, 'user': 1}
    ])


def test_values_11(mem_interactions_int_ids):
    assert check_list_equal([record for record in mem_interactions_int_ids.values(to_list=True)], [
        [1, 1, 3.0, 0],
        [2, 2, 4.0, 1],
        [3, 3, 1.0, 2],
        [1, 4, 5.0, 3],
    ])


def test_values_12(mem_interactions_int_ids):
    assert check_list_equal([record for record in mem_interactions_int_ids.values('interaction', to_list=True)],
                            [3.0, 4.0, 1.0, 5.0])


def test_values_13(mem_interactions_int_ids):
    assert check_list_equal([record for record in mem_interactions_int_ids.select('user == 2').values(to_list=True)],
                            [[2, 2, 4.0, 1]])


""" values_list """
def test_values_list_0(mem_interactions):
    assert check_list_equal(mem_interactions.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])


def test_values_list_1(mem_interactions):
    assert check_list_equal(mem_interactions.values_list(to_list=True), [
        ['jack', 'ps4', 3.0, 0],
        ['john', 'hard-drive', 4.0, 1],
        ['alfred', 'pen', 1.0, 2],
        ['jack', 'xbox', 5.0, 3],
    ])


def test_values_list_2(mem_interactions):
    assert check_list_equal(mem_interactions.values_list(columns=['item', 'user']), [
        {'item': 'ps4', 'user': 'jack'},
        {'item': 'hard-drive', 'user': 'john'},
        {'item': 'pen', 'user': 'alfred'},
        {'item': 'xbox', 'user': 'jack'}
    ])


def test_values_list_3(mem_interactions):
    assert check_list_equal(mem_interactions.values_list(columns=['item', 'user'], to_list=True), [
        ['ps4', 'jack'],
        ['hard-drive', 'john'],
        ['pen', 'alfred'],
        ['xbox', 'jack'],
    ])


def test_values_list_4(mem_interactions):
    assert check_list_equal(mem_interactions.values_list(columns=['user', 'item'], to_list=True), [
        ['jack', 'ps4'],
        ['john', 'hard-drive'],
        ['alfred', 'pen'],
        ['jack', 'xbox'],
    ])


def test_values_list_5(mem_interactions):
    try:
        mem_interactions.values_list(columns=['item', 'user', 'timestamp'])
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "timestamp".'


def test_values_list_6(mem_interactions):
    assert mem_interactions.select('interaction == 5').values_list() == [
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ]


def test_values_list_7(mem_interactions_with_iids):
    assert check_list_equal(mem_interactions_with_iids.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'uid': 0, 'iid': 0},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'uid': 2, 'iid': 2},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'uid': 0, 'iid': 3}
    ])


def test_values_list_8(mem_interactions_with_iids):
    assert check_list_equal(mem_interactions_with_iids.values_list(to_list=True), [
        ['jack', 'ps4', 3.0, 0, 0, 0],
        ['john', 'hard-drive', 4.0, 1, 1, 1],
        ['alfred', 'pen', 1.0, 2, 2, 2],
        ['jack', 'xbox', 5.0, 3, 0, 3],
    ])


def test_values_list_9(mem_interactions_with_mult_cols):
    assert check_list_equal(mem_interactions_with_mult_cols.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'timestamp': 1000.0, 'session': 5, 'tags': 'tag1;tag2'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'timestamp': 940.33, 'session': 3, 'tags': 'tag5'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'timestamp': 900.0, 'session': 2, 'tags': ''},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'timestamp': 950.52, 'session': 5, 'tags': 'tag3'}
    ])


def test_values_list_10(mem_interactions_floats):
    assert check_list_equal(mem_interactions_floats.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.2, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.1, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.5, 'rid': 3, 'user': 'jack'}
    ])


def test_values_list_11(mem_interactions_int_ids):
    assert check_list_equal(mem_interactions_int_ids.values_list(), [
        {'item': 1, 'interaction': 3.0, 'rid': 0, 'user': 1},
        {'item': 2, 'interaction': 4.0, 'rid': 1, 'user': 2},
        {'item': 3, 'interaction': 1.0, 'rid': 2, 'user': 3},
        {'item': 4, 'interaction': 5.0, 'rid': 3, 'user': 1}
    ])


""" select """
def test_select_0(mem_interactions):
    new = mem_interactions.select('interaction > 1')
    assert check_list_equal(mem_interactions.values_list(), [
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
    assert id(new) != id(mem_interactions)


def test_select_1(mem_interactions):
    same = mem_interactions.select('interaction > 1', copy=False)
    assert check_list_equal(same.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])
    assert id(same) == id(mem_interactions)


def test_select_2(mem_interactions):
    new = mem_interactions.select('interaction > 1')
    new2 = new.select('interaction < 5')
    assert check_list_equal(mem_interactions.values_list(), [
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
    assert id(new) != id(new2) and id(new2) != id(mem_interactions) and id(new) != id(mem_interactions)


def test_select_3(mem_interactions):
    same = mem_interactions.select('interaction > 1', copy=False)
    same2 = same.select('interaction < 5', copy=False)
    assert check_list_equal(same2.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'}
    ])
    assert id(same) == id(same2) and id(same2) == id(mem_interactions)


def test_select_4(mem_interactions):
    new = mem_interactions.select('interaction > 1, interaction < 5')
    assert check_list_equal(mem_interactions.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])
    assert check_list_equal(new.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'}
    ])
    assert id(new) != id(mem_interactions)


def test_select_5(mem_interactions):
    same = mem_interactions.select('interaction > 1, interaction < 5', copy=False)
    assert check_list_equal(same.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'}
    ])
    assert id(same) == id(mem_interactions)


def test_select_6(mem_interactions):
    same = mem_interactions.select('interaction > 10', copy=False)
    assert same.values_list() == []
    assert id(same) == id(mem_interactions)


def test_select_7(mem_interactions):
    new = mem_interactions.select('interaction > 1').select('interaction < 5').select('rid >= 1')
    assert check_list_equal(mem_interactions.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])
    assert new.values_list() == [
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'}
    ]
    assert id(new) != id(mem_interactions)


def test_select_8(mem_interactions):
    try:
        mem_interactions.select('interactions > 2')
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "interactions".'


def test_select_9(mem_interactions):
    try:
        mem_interactions.select('interaction >> 2')
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected operator ">>".'


def test_select_10(mem_interactions_with_iids):
    new = mem_interactions_with_iids.select('interaction > 1, interaction < 5')
    assert check_list_equal(mem_interactions_with_iids.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'uid': 0, 'iid': 0},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'uid': 2, 'iid': 2},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'uid': 0, 'iid': 3}
    ])
    assert check_list_equal(new.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'uid': 0, 'iid': 0},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1}
    ])
    assert id(new) != id(mem_interactions_with_iids)


def test_select_11(mem_interactions_with_iids):
    new = mem_interactions_with_iids.select('interaction > 1').select('interaction < 5').select('uid >= 1')
    assert check_list_equal(mem_interactions_with_iids.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'uid': 0, 'iid': 0},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'uid': 2, 'iid': 2},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'uid': 0, 'iid': 3}
    ])
    assert new.values_list() == [
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1}
    ]
    assert id(new) != id(mem_interactions_with_iids)


def test_select_12(mem_interactions_with_mult_cols):
    assert check_list_equal(mem_interactions_with_mult_cols.select('timestamp >= 950.52', copy=False).values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'timestamp': 1000.0, 'session': 5, 'tags': 'tag1;tag2'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'timestamp': 950.52, 'session': 5, 'tags': 'tag3'}
    ])


def test_select_13(mem_interactions_with_mult_cols):
    assert mem_interactions_with_mult_cols.select('timestamp >= 950.52', copy=False).select('tags == "tag3"', copy=False).values_list() == [
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'timestamp': 950.52, 'session': 5, 'tags': 'tag3'}
    ]


def test_select_14(mem_interactions_with_mult_cols):
    assert check_list_equal(mem_interactions_with_mult_cols.select('interaction != 4', copy=False).values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'timestamp': 1000.0, 'session': 5, 'tags': 'tag1;tag2'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'timestamp': 900.0, 'session': 2, 'tags': ''},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'timestamp': 950.52, 'session': 5, 'tags': 'tag3'}
    ])


def test_select_15(mem_interactions):
    try:
        mem_interactions.select('interaction > "14"')
        assert False
    except Exception as e:
        assert str(e) == 'Query "interaction > "14"" was failed to be parsed: check if no invalid comparisons are being ' \
                         'made (column of type int being compared to a str, or vice versa).'


def test_select_16(mem_interactions_floats):
    same = mem_interactions_floats.select('interaction > 1.1', copy=False)
    assert check_list_equal(same.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.2, 'rid': 1, 'user': 'john'},
        {'item': 'xbox', 'interaction': 5.5, 'rid': 3, 'user': 'jack'}
    ])
    assert id(same) == id(mem_interactions_floats)


def test_select_17(mem_interactions_floats):
    same = mem_interactions_floats.select('interaction > 1', copy=False)
    assert check_list_equal(same.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.2, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.1, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.5, 'rid': 3, 'user': 'jack'}
    ])
    assert id(same) == id(mem_interactions_floats)


def test_select_18(mem_interactions_floats):
    same = mem_interactions_floats.select('interaction >= 1.1', copy=False)
    assert check_list_equal(same.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.2, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.1, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.5, 'rid': 3, 'user': 'jack'}
    ])
    assert id(same) == id(mem_interactions_floats)


def test_select_19(mem_interactions_int_ids):
    same = mem_interactions_int_ids.select('user == 1', copy=False)
    assert check_list_equal(same.values_list(), [
        {'item': 1, 'interaction': 3.0, 'rid': 0, 'user': 1},
        {'item': 4, 'interaction': 5.0, 'rid': 3, 'user': 1},
    ])
    assert id(same) == id(mem_interactions_int_ids)


def test_select_20(mem_interactions_int_ids):
    try:
        mem_interactions_int_ids.select('user == "1"')
        assert False
    except Exception as e:
        assert str(e) == 'Query "user == "1"" was failed to be parsed: check if no invalid comparisons are being made (column of type int being compared to a str, or vice versa).'


def test_select_21(mem_interactions_int_ids):
    assert check_list_equal(mem_interactions_int_ids.select('user == 0').values_list(), [])


""" select_random_generator """
def test_select_random_generator_0(mem_interactions):
    try:
        next(mem_interactions.select_random_generator('rid > 1', seed=23))
        assert False
    except Exception as e:
        assert str(e) == 'No internal ids assigned yet.'


def test_select_random_generator_1(mem_interactions_with_iids):
    assert next(mem_interactions_with_iids.select_random_generator('rid > 1', seed=23))['rid'] == 3


def test_select_random_generator_2(mem_interactions_with_iids):
    assert next(mem_interactions_with_iids.select_random_generator('interaction > 3.0', seed=23))['interaction'] == 4.0


def test_select_random_generator_3(mem_interactions_with_iids):
    try:
        next(mem_interactions_with_iids.select_random_generator('interaction > 8.0', seed=23))
        assert False
    except Exception as e:
        assert str(e) == 'No records were found after applying the given query.'


def test_select_random_generator_4(mem_interactions_with_iids):
    try:
        next(mem_interactions_with_iids.select('interaction > 8.0').select_random_generator(seed=23))
        assert False
    except Exception as e:
        assert str(e) == 'No records were found.'


def test_select_random_generator_5(mem_interactions_with_iids):
    gen = mem_interactions_with_iids.select_random_generator('rid > 1', seed=23)
    next(gen)
    assert next(gen)['rid'] == 2


def test_select_random_generator_6(mem_interactions_with_iids):
    gen = mem_interactions_with_iids.select_random_generator(seed=23)
    assert next(gen)['rid'] == 1


def test_select_random_generator_7(mem_interactions_floats):
    mem_interactions_floats.assign_internal_ids()
    assert next(mem_interactions_floats.select_random_generator('rid > 1', seed=23))['rid'] == 3


def test_select_random_generator_8(mem_interactions_floats):
    mem_interactions_floats.assign_internal_ids()
    assert next(mem_interactions_floats.select_random_generator('interaction > 3.1', seed=23))['interaction'] == 4.2


def test_select_random_generator_9(mem_interactions_floats):
    mem_interactions_floats.assign_internal_ids()
    try:
        next(mem_interactions_floats.select_random_generator('interaction > 5.6', seed=23))
        assert False
    except Exception as e:
        assert str(e) == 'No records were found after applying the given query.'


def test_select_random_generator_10(mem_interactions_floats):
    mem_interactions_floats.assign_internal_ids()
    interaction = next(mem_interactions_floats.select_random_generator('rid > 1', seed=23))
    assert interaction['uid'] == 0
    assert isinstance(interaction['uid'], int)


def test_select_random_generator_11(mem_interactions_floats):
    mem_interactions_floats.assign_internal_ids()
    interaction = next(mem_interactions_floats.select_random_generator('rid > 1', seed=23))
    assert interaction['iid'] == 3
    assert isinstance(interaction['iid'], int)


""" null_interaction_pair_generator """
def test_null_interaction_pair_generator_0(mem_interactions):
    try:
        next(mem_interactions.null_interaction_pair_generator(seed=23))
        assert False
    except Exception as e:
        assert str(e) == 'No internal ids assigned yet.'


def test_null_interaction_pair_generator_1(mem_interactions_with_iids):
    assert next(mem_interactions_with_iids.null_interaction_pair_generator(seed=23)) == (1, 0)


def test_null_interaction_pair_generator_2(mem_interactions_with_iids):
    gen = mem_interactions_with_iids.null_interaction_pair_generator(seed=23)
    next(gen)
    assert next(gen) == (0, 2)


def test_null_interaction_pair_generator_3(mem_interactions_with_iids):
    gen = mem_interactions_with_iids.null_interaction_pair_generator(seed=23)
    next(gen), next(gen)
    assert next(gen) == (1, 3)


def test_null_interaction_pair_generator_4(mem_interactions_with_iids):
    gen = mem_interactions_with_iids.null_interaction_pair_generator(seed=23)
    next(gen), next(gen), next(gen)
    assert next(gen) == (0, 1)


def test_null_interaction_pair_generator_5(mem_interactions_with_iids):
    try:
        next(mem_interactions_with_iids.select('interaction > 8').null_interaction_pair_generator(seed=23))
        assert False
    except Exception as e:
        assert str(e) == 'No records were found.'


""" select_one """
def test_select_one_0(mem_interactions):
    assert mem_interactions.select_one('interaction > 1') == {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'}


def test_select_one_1(mem_interactions):
    assert mem_interactions.select_one('interaction > 4') == {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}


def test_select_one_2(mem_interactions):
    assert mem_interactions.select_one('rid > 0, rid < 2') == {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'}


def test_select_one_3(mem_interactions):
    assert mem_interactions.select_one('rid >= 1, interaction == 4') == {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'}


def test_select_one_4(mem_interactions):
    assert mem_interactions.select_one('rid > 0, rid < 2, interaction == 5') is None


def test_select_one_5(mem_interactions):
    try:
        mem_interactions.select_one('interaction >> 2')
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected operator ">>".'


def test_select_one_6(mem_interactions):
    try:
        mem_interactions.select_one('uid > 2')
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "uid".'


def test_select_one_7(mem_interactions_with_mult_cols):
    assert mem_interactions_with_mult_cols.select('timestamp >= 950.52', copy=False).select_one('tags == "tag3"') == \
           {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'timestamp': 950.52, 'session': 5, 'tags': 'tag3'}


def test_select_one_8(mem_interactions_with_iids):
    assert mem_interactions_with_iids.select_one('interaction > 1') == {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'uid': 0, 'iid': 0}


def test_select_one_9(mem_interactions_with_iids):
    assert mem_interactions_with_iids.select_one('interaction > 4') == {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'uid': 0, 'iid': 3}


def test_select_one_10(mem_interactions_with_iids):
    assert mem_interactions_with_iids.select_one('uid > 0, uid < 2') == {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1}


def test_select_one_11(mem_interactions_with_iids):
    assert mem_interactions_with_iids.select_one('uid > 0, uid < 2, interaction == 4') == {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1}


def test_select_one_12(mem_interactions_with_iids):
    assert mem_interactions_with_iids.select_one('uid > 0, uid < 2, interaction == 5') is None


def test_select_one_13(mem_interactions_floats):
    assert mem_interactions_floats.select_one('interaction > 4') == {'item': 'hard-drive', 'interaction': 4.2, 'rid': 1, 'user': 'john'}


def test_select_one_14(mem_interactions_floats):
    assert mem_interactions_floats.select_one('interaction > 4.2') == {'item': 'xbox', 'interaction': 5.5, 'rid': 3, 'user': 'jack'}


""" exists """
def test_exists_0(mem_interactions):
    assert mem_interactions.exists('interaction == 3') is True


def test_exists_1(mem_interactions):
    assert mem_interactions.exists('interaction == 3.0') is True


def test_exists_2(mem_interactions):
    assert mem_interactions.exists('interaction == 3.5') is False


def test_exists_3(mem_interactions):
    assert mem_interactions.exists('rid > 1') is True


def test_exists_4(mem_interactions):
    assert mem_interactions.select('rid <= 1').exists('rid > 1') is False


def test_exists_5(mem_interactions):
    assert mem_interactions.select('rid <= 1').exists('rid >= 1') is True


def test_exists_6(mem_interactions_with_mult_cols):
    assert mem_interactions_with_mult_cols.exists('timestamp >= 950.52') is True


def test_exists_7(mem_interactions_with_mult_cols):
    assert mem_interactions_with_mult_cols.exists('timestamp >= 9500.52') is False


def test_exists_8(mem_interactions_floats):
    assert mem_interactions_floats.exists('interaction == 5.0') is False


def test_exists_9(mem_interactions_floats):
    assert mem_interactions_floats.exists('interaction == 5.5') is True


def test_exists_10(mem_interactions_floats):
    assert mem_interactions_floats.exists('interaction == 3') is True


""" unique """
def test_unique_0(mem_interactions):
    assert check_list_equal(mem_interactions.unique().values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])


def test_unique_1(mem_interactions):
    assert check_list_equal(mem_interactions.unique('interaction').values_list(), [
        {'interaction': 3.0, 'rid': 0},
        {'interaction': 4.0, 'rid': 1},
        {'interaction': 1.0, 'rid': 2},
        {'interaction': 5.0, 'rid': 3}
    ])


def test_unique_2(mem_interactions):
    assert check_list_equal(mem_interactions.unique(['interaction']).values_list(), [
        {'interaction': 3.0, 'rid': 0},
        {'interaction': 4.0, 'rid': 1},
        {'interaction': 1.0, 'rid': 2},
        {'interaction': 5.0, 'rid': 3}
    ])


def test_unique_3(mem_interactions):
    assert check_list_equal(mem_interactions.unique(['user']).values_list(), [
        {'user': 'jack', 'rid': 0},
        {'user': 'john', 'rid': 1},
        {'user': 'alfred', 'rid': 2}
    ])


def test_unique_4(mem_interactions):
    assert check_list_equal(mem_interactions.unique(['user', 'interaction']).values_list(), [
        {'interaction': 3.0, 'user': 'jack', 'rid': 0},
        {'interaction': 4.0, 'user': 'john', 'rid': 1},
        {'interaction': 1.0, 'user': 'alfred', 'rid': 2},
        {'interaction': 5.0, 'user': 'jack', 'rid': 3}
    ])


def test_unique_5(mem_interactions):
    try:
        mem_interactions.unique((['user', 'test']))
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "test".'


def test_unique_6(mem_interactions_with_mult_cols):
    assert check_list_equal(mem_interactions_with_mult_cols.unique(['tags']).values_list(), [
        {'tags': 'tag1;tag2', 'rid': 0},
        {'tags': 'tag5', 'rid': 1},
        {'tags': '', 'rid': 2},
        {'tags': 'tag3', 'rid': 3}
    ])


def test_unique_7(mem_interactions_with_mult_cols):
    assert check_list_equal(mem_interactions_with_mult_cols.unique('session').values_list(), [
        {'session': 5, 'rid': 0},
        {'session': 3, 'rid': 1},
        {'session': 2, 'rid': 2},
    ])


def test_unique_8(mem_interactions_with_iids):
    assert check_list_equal(mem_interactions_with_iids.unique().values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'uid': 0, 'iid': 0},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'uid': 2, 'iid': 2},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'uid': 0, 'iid': 3}
    ])


def test_unique_9(mem_interactions_floats):
    assert check_list_equal(mem_interactions_floats.unique('interaction').values_list(), [
        {'interaction': 3.0, 'rid': 0},
        {'interaction': 4.2, 'rid': 1},
        {'interaction': 1.1, 'rid': 2},
        {'interaction': 5.5, 'rid': 3}
    ])


def test_unique_10(mem_interactions_int_ids):
    assert check_list_equal(mem_interactions_int_ids.unique('user').values_list(), [
        {'user': 1, 'rid': 0},
        {'user': 2, 'rid': 1},
        {'user': 3, 'rid': 2},
    ])


def test_unique_11(mem_interactions_int_ids):
    assert check_list_equal(mem_interactions_int_ids.unique('item').values_list(), [
        {'item': 1, 'rid': 0},
        {'item': 2, 'rid': 1},
        {'item': 3, 'rid': 2},
        {'item': 4, 'rid': 3},
    ])


""" count_unique """
def test_count_unique_0(mem_interactions):
    assert mem_interactions.count_unique(['user']) == 3


def test_count_unique_1(mem_interactions_with_mult_cols):
    assert mem_interactions_with_mult_cols.count_unique('session') == 3


def test_count_unique_2(mem_interactions):
    try:
        mem_interactions.count_unique((['user', 'test']))
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "test".'


def test_count_unique_3(mem_interactions):
    assert mem_interactions.count_unique(['user', 'interaction']) == 4


""" max """
def test_max_0(mem_interactions):
    assert mem_interactions.max('interaction') == 5.0


def test_max_1(mem_interactions):
    assert mem_interactions.max('user') == 'john'


def test_max_2(mem_interactions_with_mult_cols):
    assert mem_interactions_with_mult_cols.max('session') == 5


def test_max_3(mem_interactions):
    try:
        mem_interactions.max((['user']))
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column type "<class \'list\'>".'


def test_max_4(mem_interactions):
    try:
        mem_interactions.max('users')
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "users".'


def test_max_5(mem_interactions):
    try:
        mem_interactions.max()
        assert False
    except Exception as e:
        assert str(e) == 'No column was given.'


def test_max_6(mem_interactions_with_iids):
    assert mem_interactions_with_iids.max('uid') == 2


def test_max_7(mem_interactions_floats):
    assert mem_interactions_floats.max('interaction') == 5.5


def test_max_8(mem_interactions):
    assert mem_interactions.max('rid') == 3


def test_max_9(mem_df):
    assert InteractionDataset.read_df(mem_df, user_label='u', item_label='i', interaction_label='r').max('rid') == 3


""" min """
def test_min_0(mem_interactions):
    assert mem_interactions.min('interaction') == 1.0


def test_min_1(mem_interactions):
    assert mem_interactions.min('user') == 'alfred'


def test_min_2(mem_interactions_with_mult_cols):
    assert mem_interactions_with_mult_cols.min('session') == 2


def test_min_3(mem_interactions):
    try:
        mem_interactions.min((['user']))
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column type "<class \'list\'>".'


def test_min_4(mem_interactions):
    try:
        mem_interactions.min('users')
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "users".'


def test_min_5(mem_interactions):
    try:
        mem_interactions.min()
        assert False
    except Exception as e:
        assert str(e) == 'No column was given.'


def test_min_6(mem_interactions_with_iids):
    assert mem_interactions_with_iids.min('uid') == 0


def test_min_7(mem_interactions_floats):
    assert mem_interactions_floats.min('interaction') == 1.1


def test_min_8(mem_df):
    assert InteractionDataset.read_df(mem_df, user_label='u', item_label='i', interaction_label='r').min('rid') == 0


""" select_user_interaction_vec """
def test_select_user_interaction_vec_0(mem_interactions):
    try:
        mem_interactions.select_user_interaction_vec(0).toarray().ravel()
        assert False
    except Exception as e:
        assert str(e) == 'Cannot retrieve user interaction vector without assigned internal ids.'


def test_select_user_interaction_vec_1(mem_interactions):
    mem_interactions = mem_interactions.select('interaction > 2.5', copy=False)
    mem_interactions.assign_internal_ids()
    assert np.array_equal(mem_interactions.select_user_interaction_vec(0).toarray().ravel(), [3.0, 0, 5.0])
    assert np.array_equal(mem_interactions.select_user_interaction_vec(1).toarray().ravel(), [0, 4.0, 0])


def test_select_user_interaction_vec_2(mem_interactions):
    new = mem_interactions.select('interaction > 2.5').select('interaction < 5')
    new.assign_internal_ids()
    assert np.array_equal(new.select_user_interaction_vec(0).toarray().ravel(), [3.0, 0])
    assert np.array_equal(new.select_user_interaction_vec(1).toarray().ravel(), [0, 4.0])
    try:
        new.select_user_interaction_vec(2)
        assert False
    except Exception as e:
        assert str(e) == 'User internal id 2 was not found.'


def test_select_user_interaction_vec_3(mem_interactions_with_iids):
    assert np.array_equal(mem_interactions_with_iids.select_user_interaction_vec(0).toarray().ravel(), [3.0, 0, 0, 5.0])


def test_select_user_interaction_vec_4(mem_interactions_with_iids):
    assert np.array_equal(mem_interactions_with_iids.select_user_interaction_vec(1).toarray().ravel(), [0, 4.0, 0, 0])


def test_select_user_interaction_vec_5(mem_interactions_with_iids):
    assert np.array_equal(mem_interactions_with_iids.select_user_interaction_vec(2).toarray().ravel(), [0, 0, 1.0, 0])


def test_select_user_interaction_vec_6(mem_interactions_with_iids):
    try:
        mem_interactions_with_iids.select_user_interaction_vec(3)
        assert False
    except Exception as e:
        assert str(e) == 'User internal id 3 was not found.'


def test_select_user_interaction_vec_7(mem_interactions_floats):
    mem_interactions_floats.assign_internal_ids()
    assert np.array_equal(mem_interactions_floats.select_user_interaction_vec(0).toarray().ravel(), [3.0, 0, 0, 5.5])


def test_select_user_interaction_vec_8(mem_interactions_floats):
    mem_interactions_floats.assign_internal_ids()
    assert np.array_equal(mem_interactions_floats.select_user_interaction_vec(1).toarray().ravel(), [0, 4.2, 0, 0])


def test_select_user_interaction_vec_9(mem_interactions_floats):
    mem_interactions_floats.assign_internal_ids()
    assert np.array_equal(mem_interactions_floats.select_user_interaction_vec(2).toarray().ravel(), [0, 0, 1.1, 0])


def test_select_user_interaction_vec_10(mem_interactions):
    """Test when memory error occurs and users_records is not defined."""
    def throw(_): raise MemoryError
    mem_interactions = mem_interactions.select('interaction > 2.5', copy=False)
    mem_interactions.assign_internal_ids()
    mem_interactions._users_records = None
    func_type = type(mem_interactions._build_interaction_matrix)
    mem_interactions._build_interaction_matrix = func_type(throw, mem_interactions)
    assert np.array_equal(mem_interactions.select_user_interaction_vec(0).toarray().ravel(), [3.0, 0, 5.0])
    assert np.array_equal(mem_interactions.select_user_interaction_vec(1).toarray().ravel(), [0, 4.0, 0])


def test_select_user_interaction_vec_11(mem_interactions):
    """Test when memory error occurs"""
    def throw(_): raise MemoryError
    mem_interactions = mem_interactions.select('interaction > 2.5', copy=False)
    mem_interactions.assign_internal_ids()
    func_type = type(mem_interactions._build_interaction_matrix)
    mem_interactions._build_interaction_matrix = func_type(throw, mem_interactions)
    assert np.array_equal(mem_interactions.select_user_interaction_vec(0).toarray().ravel(), [3.0, 0, 5.0])
    assert np.array_equal(mem_interactions.select_user_interaction_vec(1).toarray().ravel(), [0, 4.0, 0])


""" select_item_interaction_vec """
def test_select_item_interaction_vec_0(mem_interactions):
    try:
        mem_interactions.select_item_interaction_vec(0).toarray().ravel()
        assert False
    except Exception as e:
        assert str(e) == 'Cannot retrieve user interaction vector without assigned internal ids.'


def test_select_item_interaction_vec_1(mem_interactions):
    mem_interactions = mem_interactions.select('interaction > 2.5', copy=False)
    mem_interactions.assign_internal_ids()
    assert np.array_equal(mem_interactions.select_item_interaction_vec(0).toarray().ravel(), [3.0, 0])
    assert np.array_equal(mem_interactions.select_item_interaction_vec(1).toarray().ravel(), [0, 4.0])


def test_select_item_interaction_vec_2(mem_interactions):
    new = mem_interactions.select('interaction > 2.5').select('interaction < 4')
    new.assign_internal_ids()
    assert np.array_equal(new.select_item_interaction_vec(0).toarray().ravel(), [3.0])
    try:
        new.select_item_interaction_vec(1)
        assert False
    except Exception as e:
        assert str(e) == 'Item internal id 1 was not found.'


def test_select_item_interaction_vec_3(mem_interactions_with_iids):
    assert np.array_equal(mem_interactions_with_iids.select_item_interaction_vec(0).toarray().ravel(), [3.0, 0, 0])


def test_select_item_interaction_vec_4(mem_interactions_with_iids):
    assert np.array_equal(mem_interactions_with_iids.select_item_interaction_vec(1).toarray().ravel(), [0, 4.0, 0])


def test_select_item_interaction_vec_5(mem_interactions_with_iids):
    assert np.array_equal(mem_interactions_with_iids.select_item_interaction_vec(2).toarray().ravel(), [0, 0, 1.0])


def test_select_item_interaction_vec_6(mem_interactions_with_iids):
    assert np.array_equal(mem_interactions_with_iids.select_item_interaction_vec(3).toarray().ravel(), [5.0, 0, 0])


def test_select_item_interaction_vec_7(mem_interactions_with_iids):
    try:
        mem_interactions_with_iids.select_item_interaction_vec(4)
        assert False
    except Exception as e:
        assert str(e) == 'Item internal id 4 was not found.'


def test_select_item_interaction_vec_8(mem_interactions_floats):
    mem_interactions_floats.assign_internal_ids()
    assert np.array_equal(mem_interactions_floats.select_item_interaction_vec(0).toarray().ravel(), [3.0, 0, 0])


def test_select_item_interaction_vec_9(mem_interactions_floats):
    mem_interactions_floats.assign_internal_ids()
    assert np.array_equal(mem_interactions_floats.select_item_interaction_vec(1).toarray().ravel(), [0, 4.2, 0])


def test_select_item_interaction_vec_10(mem_interactions_floats):
    mem_interactions_floats.assign_internal_ids()
    assert np.array_equal(mem_interactions_floats.select_item_interaction_vec(2).toarray().ravel(), [0, 0, 1.1])


def test_select_item_interaction_vec_11(mem_interactions_floats):
    mem_interactions_floats.assign_internal_ids()
    assert np.array_equal(mem_interactions_floats.select_item_interaction_vec(3).toarray().ravel(), [5.5, 0, 0])


def test_select_user_interaction_vec_12(mem_interactions):
    """Test when memory error occurs and items_records is not defined."""
    def throw(_): raise MemoryError
    mem_interactions = mem_interactions.select('interaction > 2.5', copy=False)
    mem_interactions.assign_internal_ids()
    mem_interactions._items_records = None
    func_type = type(mem_interactions._build_interaction_matrix)
    mem_interactions._build_interaction_matrix = func_type(throw, mem_interactions)
    assert np.array_equal(mem_interactions.select_item_interaction_vec(0).toarray().ravel(), [3.0, 0])
    assert np.array_equal(mem_interactions.select_item_interaction_vec(1).toarray().ravel(), [0, 4.0])


def test_select_user_interaction_vec_13(mem_interactions):
    """Test when memory error occurs"""
    def throw(_): raise MemoryError
    mem_interactions = mem_interactions.select('interaction > 2.5', copy=False)
    mem_interactions.assign_internal_ids()
    func_type = type(mem_interactions._build_interaction_matrix)
    mem_interactions._build_interaction_matrix = func_type(throw, mem_interactions)
    assert np.array_equal(mem_interactions.select_item_interaction_vec(0).toarray().ravel(), [3.0, 0])
    assert np.array_equal(mem_interactions.select_item_interaction_vec(1).toarray().ravel(), [0, 4.0])


""" user_to_uid """
def test_user_to_uid_0(mem_interactions_with_iids):
    assert mem_interactions_with_iids.user_to_uid('jack') == 0
    assert mem_interactions_with_iids.user_to_uid('john') == 1
    assert mem_interactions_with_iids.user_to_uid('alfred') == 2


def test_user_to_uid_1(mem_interactions_with_iids):
    assert mem_interactions_with_iids.user_to_uid('bla') is None


def test_user_to_uid_2(mem_interactions_with_iids):
    assert mem_interactions_with_iids.select('user == "alfred"').user_to_uid('alfred') == 2


def test_user_to_uid_3(mem_interactions_with_iids):
    assert mem_interactions_with_iids.select('user == "alfred"').user_to_uid('jack') == 0


def test_user_to_uid_4(mem_interactions_with_iids):
    assert mem_interactions_with_iids.select('rid > 2').user_to_uid('jack') == 0


def test_user_to_uid_5(mem_interactions):
    try:
        mem_interactions.user_to_uid('jack')
        assert False
    except Exception as e:
        assert str(e) == 'No internal ids assigned yet.'


def test_user_to_uid_6(mem_interactions_int_ids):
    mem_interactions_int_ids.assign_internal_ids()
    assert mem_interactions_int_ids.user_to_uid(1) == 0
    assert mem_interactions_int_ids.user_to_uid(2) == 1
    assert mem_interactions_int_ids.user_to_uid(3) == 2


def test_user_to_uid_7(mem_interactions_int_ids):
    mem_interactions_int_ids.assign_internal_ids()
    try:
        mem_interactions_int_ids.user_to_uid('jack')
        assert False
    except Exception as e:
        assert str(e) == "The provided user type does not match the inferred type (expected: int, found: <class 'str'>"


""" uid_to_user """
def test_uid_to_user_0(mem_interactions_with_iids):
    assert mem_interactions_with_iids.uid_to_user(0) == 'jack'
    assert mem_interactions_with_iids.uid_to_user(1) == 'john'
    assert mem_interactions_with_iids.uid_to_user(2) == 'alfred'


def test_uid_to_user_1(mem_interactions_with_iids):
    assert mem_interactions_with_iids.uid_to_user(3) is None


def test_uid_to_user_2(mem_interactions_with_iids):
    assert mem_interactions_with_iids.select('user == "alfred"').uid_to_user(0) == 'jack'


def test_uid_to_user_3(mem_interactions_with_iids):
    assert mem_interactions_with_iids.select('user == "alfred"').uid_to_user(1) == 'john'


def test_uid_to_user_4(mem_interactions_with_iids):
    assert mem_interactions_with_iids.select('rid > 2').uid_to_user(1) == 'john'


def test_uid_to_user_5(mem_interactions):
    try:
        mem_interactions.uid_to_user(0)
        assert False
    except Exception as e:
        assert str(e) == 'No internal ids assigned yet.'


def test_uid_to_user_6(mem_interactions_int_ids):
    mem_interactions_int_ids.assign_internal_ids()
    assert mem_interactions_int_ids.uid_to_user(0) == 1
    assert mem_interactions_int_ids.uid_to_user(1) == 2
    assert mem_interactions_int_ids.uid_to_user(2) == 3


""" item_to_iid """
def test_item_to_iid_0(mem_interactions_with_iids):
    assert mem_interactions_with_iids.item_to_iid('ps4') == 0
    assert mem_interactions_with_iids.item_to_iid('hard-drive') == 1
    assert mem_interactions_with_iids.item_to_iid('pen') == 2
    assert mem_interactions_with_iids.item_to_iid('xbox') == 3


def test_item_to_iid_1(mem_interactions_with_iids):
    assert mem_interactions_with_iids.item_to_iid('bla') is None


def test_item_to_iid_2(mem_interactions_with_iids):
    assert mem_interactions_with_iids.select('item == "hard-drive"').item_to_iid('hard-drive') == 1


def test_item_to_iid_3(mem_interactions_with_iids):
    assert mem_interactions_with_iids.select('item == "hard-drive"').item_to_iid('pen') == 2


def test_item_to_iid_4(mem_interactions_with_iids):
    assert mem_interactions_with_iids.select('rid > 2').item_to_iid('xbox') == 3


def test_item_to_iid_5(mem_interactions):
    try:
        mem_interactions.item_to_iid('ps4')
        assert False
    except Exception as e:
        assert str(e) == 'No internal ids assigned yet.'


def test_item_to_iid_6(mem_interactions_int_ids):
    mem_interactions_int_ids.assign_internal_ids()
    assert mem_interactions_int_ids.item_to_iid(1) == 0
    assert mem_interactions_int_ids.item_to_iid(2) == 1
    assert mem_interactions_int_ids.item_to_iid(3) == 2
    assert mem_interactions_int_ids.item_to_iid(4) == 3


def test_item_to_iid_7(mem_interactions_int_ids):
    mem_interactions_int_ids.assign_internal_ids()
    try:
        mem_interactions_int_ids.item_to_iid('ps4')
        assert False
    except Exception as e:
        assert str(e) == "The provided item type does not match the inferred type (expected: int, found: <class 'str'>"


""" iid_to_item """
def test_iid_to_item_0(mem_interactions_with_iids):
    assert mem_interactions_with_iids.iid_to_item(0) == 'ps4'
    assert mem_interactions_with_iids.iid_to_item(1) == 'hard-drive'
    assert mem_interactions_with_iids.iid_to_item(2) == 'pen'
    assert mem_interactions_with_iids.iid_to_item(3) == 'xbox'


def test_iid_to_item_1(mem_interactions_with_iids):
    assert mem_interactions_with_iids.iid_to_item(4) is None


def test_iid_to_item_2(mem_interactions_with_iids):
    assert mem_interactions_with_iids.select('iid == 0').iid_to_item(0) == 'ps4'


def test_iid_to_item_3(mem_interactions_with_iids):
    assert mem_interactions_with_iids.select('iid == 0').iid_to_item(1) == 'hard-drive'


def test_iid_to_item_4(mem_interactions_with_iids):
    assert mem_interactions_with_iids.select('rid > 1').iid_to_item(3) == 'xbox'


def test_iid_to_item_5(mem_interactions):
    try:
        mem_interactions.iid_to_item(0)
        assert False
    except Exception as e:
        assert str(e) == 'No internal ids assigned yet.'


def test_iid_to_item_6(mem_interactions_int_ids):
    mem_interactions_int_ids.assign_internal_ids()
    assert mem_interactions_int_ids.iid_to_item(0) == 1
    assert mem_interactions_int_ids.iid_to_item(1) == 2
    assert mem_interactions_int_ids.iid_to_item(2) == 3
    assert mem_interactions_int_ids.iid_to_item(3) == 4


""" drop """
def test_drop_0(mem_interactions):
    assert check_list_equal(mem_interactions.drop([0, 2]).values_list(), [
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])


def test_drop_1(mem_interactions):
    assert mem_interactions.drop([0, 2]).select('interaction > 4').values_list() == [
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ]


def test_drop_2(mem_interactions):
    assert mem_interactions.select('interaction > 4').drop([0, 2]).values_list() == [
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ]


def test_drop_3(mem_interactions):
    assert mem_interactions.select('interaction > 1', copy=False).select('interaction > 2', copy=False)\
               .drop([0]).drop([3]).drop([0]).values_list() == [
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
    ]


def test_drop_4(mem_interactions_with_mult_cols):
    assert mem_interactions_with_mult_cols.select('interaction > 1', copy=False).select('interaction > 2', copy=False)\
               .drop([0]).drop([3]).drop([0]).values_list() == [
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'timestamp': 940.33, 'session': 3, 'tags': 'tag5'},
    ]


def test_drop_5(mem_interactions):
    assert check_list_equal(mem_interactions.drop([0, 2], keep=True).values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
    ])


""" apply """
def test_apply_0(mem_interactions):
    mem_interactions.apply('interaction', lambda x: 1 if x > 2.5 else 0)
    assert check_list_equal([record for record in mem_interactions.values()], [
        {'item': 'ps4', 'interaction': 1, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 1, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 1, 'rid': 3, 'user': 'jack'}
    ])


def test_apply_1(mem_interactions):
    mem_interactions.apply('interaction', lambda x: x / 5)
    assert check_list_equal([record for record in mem_interactions.values()], [
        {'item': 'ps4', 'interaction': 0.6, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 0.8, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 0.2, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 1, 'rid': 3, 'user': 'jack'}
    ])


def test_apply_2(mem_interactions_with_mult_cols):
    mem_interactions_with_mult_cols.apply('tags', hash)
    assert check_list_equal(mem_interactions_with_mult_cols.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'timestamp': 1000.0, 'session': 5,  'tags': hash('tag1;tag2')},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'timestamp': 940.33, 'session': 3, 'tags': hash('tag5')},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'timestamp': 900.0, 'session': 2, 'tags': hash('')},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'timestamp': 950.52, 'session': 5, 'tags': hash('tag3')}
    ])


def test_apply_3(mem_interactions):
    try:
        mem_interactions.apply('bla', lambda x: x)
        assert False
    except Exception as e:
        assert str(e) == 'Unexpected column "bla".'


def test_apply_4(mem_interactions):
    try:
        mem_interactions.apply('rid', lambda x: x)
        assert False
    except Exception as e:
        assert str(e) == 'Column "rid" is read-only.'


def test_apply_5(mem_interactions):
    mem_interactions.assign_internal_ids()
    try:
        mem_interactions.apply('uid', lambda x: x)
        assert False
    except Exception as e:
        assert str(e) == 'Column "uid" is read-only.'


def test_apply_6(mem_interactions):
    mem_interactions.assign_internal_ids()
    try:
        mem_interactions.apply('iid', lambda x: x)
        assert False
    except Exception as e:
        assert str(e) == 'Column "iid" is read-only.'


def test_apply_7(mem_interactions):
    mem_interactions.assign_internal_ids()
    try:
        mem_interactions.apply('user', lambda x: x)
        assert False
    except Exception as e:
        assert str(e) == 'Column "user" is read-only.'


def test_apply_8(mem_interactions):
    mem_interactions.assign_internal_ids()
    try:
        mem_interactions.apply('item', lambda x: x)
        assert False
    except Exception as e:
        assert str(e) == 'Column "item" is read-only.'


def test_apply_9(mem_interactions):
    try:
        mem_interactions.apply('interaction', lambda x: x/0)
        assert False
    except Exception as e:
        assert str(e) == 'Failed to apply operation on column "interaction". Details: division by zero'


def test_apply_10(mem_interactions):
    try:
        mem_interactions.apply('interaction', None)
        assert False
    except Exception as e:
        assert str(e) == 'Failed to apply operation on column "interaction". Details: \'NoneType\' object is not callable'


def test_apply_11(mem_interactions_with_mult_cols):
    try:
        mem_interactions_with_mult_cols.apply('tags', lambda x: x.split())
        assert False
    except Exception as e:
        assert str(e) == 'Failed to apply operation on column "tags". Details: New column type "<class \'list\'>" is not supported. Supported types: [<class \'int\'>, <class \'float\'>, <class \'str\'>].'


def test_apply_12(mem_interactions):
    mem_interactions_old = mem_interactions.copy()
    mem_interactions.apply('interaction', lambda x: 1 if x > 2.5 else 0)
    assert check_list_equal([record for record in mem_interactions.values()], [
        {'item': 'ps4', 'interaction': 1, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 1, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 1, 'rid': 3, 'user': 'jack'}
    ])
    assert check_list_equal([record for record in mem_interactions_old.values()], [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])


def test_apply_13(mem_interactions):
    mem_interactions.assign_internal_ids()

    mem_interactions.apply('interaction', lambda x: 1 if x > 2.5 else 0)
    # must have not built unused structures
    assert mem_interactions._cached_interaction_matrix is None
    assert  mem_interactions._cached_trans_interaction_matrix is None


def test_apply_14(mem_interactions):
    mem_interactions.assign_internal_ids()
    mem_interactions._build_interaction_matrix()

    # save previous state
    saved_cached_int_matrix = mem_interactions._cached_interaction_matrix
    saved_cached_trans_int_matrix = mem_interactions._cached_trans_interaction_matrix

    mem_interactions.apply('interaction', lambda x: 1 if x > 2.5 else 0)
    # must update the cached interaction matrix
    assert id(saved_cached_int_matrix) != id(mem_interactions._cached_interaction_matrix)
    assert id(saved_cached_trans_int_matrix) != id(mem_interactions._cached_trans_interaction_matrix)


""" save """
def test_save_0(mem_interactions):
    mem_interactions.save('testtmp.csv', write_header=True)
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


def test_save_1(mem_interactions_with_iids):
    mem_interactions_with_iids.save('testtmp.csv')
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


def test_save_2(mem_interactions):
    mem_interactions.save('testtmp.csv')
    try:
        assert check_list_equal(InteractionDataset('testtmp.csv', columns=['user', 'item', 'interaction'], in_memory=IN_MEMORY).values_list(), [
            {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
            {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
            {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
            {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
        ])
    finally:
        os.remove('testtmp.csv')


def test_save_3(mem_interactions_with_iids):
    mem_interactions_with_iids.save('testtmp.csv')
    try:
        assert check_list_equal(InteractionDataset('testtmp.csv', columns=['user', 'item', 'interaction'], in_memory=IN_MEMORY).values_list(), [
            {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
            {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
            {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
            {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
        ])
    finally:
        os.remove('testtmp.csv')


def test_save_4(mem_interactions_with_iids):
    mem_interactions_with_iids.save('testtmp.csv')
    try:
        assert check_list_equal(InteractionDataset('testtmp.csv', columns=['user', 'item', 'interaction'], in_memory=IN_MEMORY).values_list(), [
            {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
            {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
            {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
            {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
        ])
    finally:
        os.remove('testtmp.csv')


def test_save_5(mem_interactions_with_iids):
    mem_interactions_with_iids.select('interaction > 3').save('testtmp.csv')
    try:
        assert check_list_equal(InteractionDataset('testtmp.csv', columns=['user', 'item', 'interaction'], in_memory=IN_MEMORY).values_list(), [
            {'item': 'hard-drive', 'interaction': 4.0, 'rid': 0, 'user': 'john'},
            {'item': 'xbox', 'interaction': 5.0, 'rid': 1, 'user': 'jack'}
        ])
    finally:
        os.remove('testtmp.csv')


def test_save_6(mem_interactions_with_mult_cols):
    mem_interactions_with_mult_cols.save('testtmp.csv')
    try:
        assert check_list_equal(InteractionDataset('testtmp.csv', columns=['user', 'item', 'interaction', 'timestamp', 'session', 'tags'], in_memory=IN_MEMORY).values_list(), [
            {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'timestamp': 1000.0, 'session': 5, 'tags': 'tag1;tag2'},
            {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'timestamp': 940.33, 'session': 3, 'tags': 'tag5'},
            {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'timestamp': 900.0, 'session': 2, 'tags': ''},
            {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'timestamp': 950.52, 'session': 5, 'tags': 'tag3'}
        ])
    finally:
        os.remove('testtmp.csv')


def test_save_7(mem_interactions_with_mult_cols):
    mem_interactions_with_mult_cols.save('testtmp.csv')
    try:
        assert check_list_equal(InteractionDataset('testtmp.csv', columns=['user', 'item', 'interaction', 'timestamp', None, 'tags'], in_memory=IN_MEMORY).values_list(), [
            {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'timestamp': 1000.0, 'tags': 'tag1;tag2'},
            {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'timestamp': 940.33, 'tags': 'tag5'},
            {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'timestamp': 900.0, 'tags': ''},
            {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'timestamp': 950.52, 'tags': 'tag3'}
        ])
    finally:
        os.remove('testtmp.csv')


def test_save_8(mem_interactions_floats):
    mem_interactions_floats.save('testtmp.csv', write_header=True)
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


def test_save_9(mem_interactions_int_ids):
    mem_interactions_int_ids.save('testtmp.csv', write_header=True)
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
def test_assign_internal_ids_0(mem_interactions):
    assert mem_interactions.has_internal_ids is False


def test_assign_internal_ids_1(mem_interactions):
    mem_interactions.assign_internal_ids()
    assert mem_interactions.has_internal_ids is True


def test_assign_internal_ids_2(mem_interactions_with_iids):
    assert mem_interactions_with_iids.has_internal_ids is True


def test_assign_internal_ids_3(mem_interactions):
    mem_interactions.assign_internal_ids()
    assert check_list_equal(mem_interactions.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack', 'uid': 0, 'iid': 0},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john', 'uid': 1, 'iid': 1},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred', 'uid': 2, 'iid': 2},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack', 'uid': 0, 'iid': 3}
    ])


""" remove_internal_ids """
def test_remove_internal_ids_0(mem_interactions):
    mem_interactions.assign_internal_ids()
    mem_interactions.remove_internal_ids()
    assert mem_interactions.has_internal_ids is False


def test_remove_internal_ids_1(mem_interactions_with_iids):
    mem_interactions_with_iids.remove_internal_ids()
    assert mem_interactions_with_iids.has_internal_ids is False


def test_remove_internal_ids_2(mem_interactions_with_iids):
    mem_interactions_with_iids.remove_internal_ids()
    print(mem_interactions_with_iids.values_list())
    assert check_list_equal(mem_interactions_with_iids.values_list(), [
        {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
        {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
        {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
        {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
    ])


""" Specific tests """
""" read_df """
def test_read_df_0(mem_df):
    InteractionDataset.read_df(mem_df, user_label='u', item_label='i', interaction_label='r').values_list() # keep this to test if we do not break a pandas df when reading from it
    assert check_list_equal(InteractionDataset.read_df(mem_df, user_label='u', item_label='i', interaction_label='r').values_list(), [
            {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
            {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
            {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
            {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
        ])


def test_read_df_1(mem_df_with_columns):
    assert check_list_equal(InteractionDataset.read_df(mem_df_with_columns).values_list(), [
            {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
            {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
            {'item': 'pen', 'interaction': 1.0, 'rid': 2, 'user': 'alfred'},
            {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
        ])


def test_read_df_2(mem_df_with_columns):
    assert check_list_equal(InteractionDataset.read_df(mem_df_with_columns).select('interaction > 3').values_list(), [
            {'item': 'hard-drive', 'interaction': 4.0, 'rid': 1, 'user': 'john'},
            {'item': 'xbox', 'interaction': 5.0, 'rid': 3, 'user': 'jack'}
        ])


def test_read_df_3(mem_df_floats):
    assert check_list_equal(InteractionDataset.read_df(mem_df_floats, user_label='u', item_label='i', interaction_label='r').values_list(), [
            {'item': 'ps4', 'interaction': 3.0, 'rid': 0, 'user': 'jack'},
            {'item': 'hard-drive', 'interaction': 4.2, 'rid': 1, 'user': 'john'},
            {'item': 'pen', 'interaction': 1.1, 'rid': 2, 'user': 'alfred'},
            {'item': 'xbox', 'interaction': 5.5, 'rid': 3, 'user': 'jack'}
        ])


def test_read_df_4(mem_df_int_ids):
    assert check_list_equal(InteractionDataset.read_df(mem_df_int_ids, user_label='u', item_label='i', interaction_label='r').values_list(), [
            {'item': 1, 'interaction': 3.0, 'rid': 0, 'user': 1},
            {'item': 2, 'interaction': 4.0, 'rid': 1, 'user': 2},
            {'item': 3, 'interaction': 1.0, 'rid': 2, 'user': 3},
            {'item': 4, 'interaction': 5.0, 'rid': 3, 'user': 1}
        ])
