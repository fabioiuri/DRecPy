from DRecPy.Dataset import available_datasets
from DRecPy.Dataset import get_full_dataset
from DRecPy.Dataset import get_test_dataset
from DRecPy.Dataset import get_train_dataset
from DRecPy.Dataset import MemoryInteractionDataset
from DRecPy.Dataset import DatabaseInteractionDataset
import numpy as np
import shutil
import os

""" available_datasets """


def test_available_datasets_0():
    assert np.array_equal(available_datasets(), ['ml-100k', 'ml-1m', 'ml-10m', 'ml-20m', 'bx'])


""" get_full_dataset """


def test_get_full_dataset_0():
    try:
        get_full_dataset('')
    except FileNotFoundError as e:
        assert str(e) == '"" is not a valid dataset. Supported datasets: ml-100k, ml-1m, ml-10m, ml-20m, bx.'


def test_get_full_dataset_1():
    try:
        shutil.rmtree(os.path.join(os.path.expanduser('~') + '/.DRecPy_data/', 'ml-100k'))
    except FileNotFoundError:
        pass

    ret = get_full_dataset('ml-100k')
    assert open(os.path.expanduser('~') + '/.DRecPy_data/ml-100k/ua.base', 'r') is not None
    assert (len(ret), len(ret.columns)) == (100000, 5)
    assert next(ret.values()) == {'interaction': 3, 'user': 196, 'item': 242, 'rid': 0, 'timestamp': 881250949}


def test_get_full_dataset_2():
    ret = get_full_dataset('ml-100k')
    assert isinstance(ret, MemoryInteractionDataset)


def test_get_full_dataset_3():
    ret = get_full_dataset('ml-100k', force_out_of_memory=True)
    assert isinstance(ret, DatabaseInteractionDataset)


def test_get_full_dataset_4():
    try:
        os.environ['DATA_FOLDER'] = os.path.curdir

        try:
            shutil.rmtree(os.path.join(os.environ.get('DATA_FOLDER'), 'ml-100k'))
        except FileNotFoundError:
            pass

        ret = get_full_dataset('ml-100k')
        assert open(os.path.curdir + '/ml-100k/ua.base', 'r') is not None
        assert (len(ret), len(ret.columns)) == (100000, 5)
        assert next(ret.values()) == {'interaction': 3, 'user': 196, 'item': 242, 'rid': 0, 'timestamp': 881250949}
        shutil.rmtree(os.path.join(os.environ.get('DATA_FOLDER'), 'ml-100k'))
    finally:
        del os.environ['DATA_FOLDER']


def test_get_full_dataset_5():
    try:
        shutil.rmtree(os.path.join(os.path.expanduser('~') + '/.DRecPy_data/', 'bx'))
    except FileNotFoundError:
        pass

    ret = get_full_dataset('bx')
    assert open(os.path.expanduser('~') + '/.DRecPy_data/bx/BX-Book-Ratings.csv', 'r') is not None
    assert (len(ret), len(ret.columns)) == (1149780, 4)
    assert next(ret.values()) == {'interaction': 0, 'user': 276725, 'item': '034545104X', 'rid': 0}


def test_get_full_dataset_6():
    try:
        shutil.rmtree(os.path.join(os.path.expanduser('~') + '/.DRecPy_data/', 'ml-1m'))
    except FileNotFoundError:
        pass

    ret = get_full_dataset('ml-1m')
    assert open(os.path.expanduser('~') + '/.DRecPy_data/ml-1m/ratings.dat', 'r') is not None
    assert (len(ret), len(ret.columns)) == (1000209, 5)
    assert next(ret.values()) == {'interaction': 5, 'user': 1, 'item': 1193, 'rid': 0, 'timestamp': 978300760}


def test_get_full_dataset_7():
    try:
        shutil.rmtree(os.path.join(os.path.expanduser('~') + '/.DRecPy_data/', 'ml-10m'))
    except FileNotFoundError:
        pass

    ret = get_full_dataset('ml-10m')
    assert open(os.path.expanduser('~') + '/.DRecPy_data/ml-10m/ratings.dat', 'r') is not None
    assert (len(ret), len(ret.columns)) == (10000054, 5)
    assert next(ret.values()) == {'interaction': 5, 'user': 1, 'item': 122, 'rid': 0, 'timestamp': 838985046}


def test_get_full_dataset_8():
    try:
        shutil.rmtree(os.path.join(os.path.expanduser('~') + '/.DRecPy_data/', 'ml-20m'))
    except FileNotFoundError:
        pass

    ret = get_full_dataset('ml-20m')

    assert open(os.path.expanduser('~') + '/.DRecPy_data/ml-20m/ratings.csv', 'r') is not None
    assert (len(ret), len(ret.columns)) == (20000263, 5)
    assert next(ret.values()) == {'interaction': 3.5, 'user': 1, 'item': 2, 'rid': 0, 'timestamp': 1112486027}


""" get_test_dataset """


def test_get_test_dataset_0():
    try:
        get_test_dataset('')
    except FileNotFoundError as e:
        assert str(e) == '"" is not a valid dataset. Supported datasets: ml-100k, ml-1m, ml-10m, ml-20m, bx.'


def test_get_test_dataset_1():
    ret = get_test_dataset('ml-100k')
    assert (len(ret), len(ret.columns)) == (9430, 5)
    assert next(ret.values(columns=['interaction', 'item', 'user', 'timestamp'])) == \
           {'interaction': 4, 'user': 1, 'item': 20, 'timestamp': 887431883}


def test_get_test_dataset_2():
    ret = get_test_dataset('ml-100k')
    assert isinstance(ret, MemoryInteractionDataset)


def test_get_test_dataset_3():
    ret = get_test_dataset('ml-100k', force_out_of_memory=True)
    assert isinstance(ret, DatabaseInteractionDataset)


def test_get_test_dataset_4():
    ret = get_test_dataset('bx')
    assert (len(ret), len(ret.columns)) == (120530, 4)
    assert next(ret.values(columns=['interaction', 'item', 'user'])) == \
           {'interaction': 0, 'item': '034544003X', 'user': 276762}


def test_get_test_dataset_5():
    ret = get_test_dataset('ml-1m')
    assert (len(ret), len(ret.columns)) == (60400, 5)
    assert next(ret.values(columns=['interaction', 'item', 'user', 'timestamp'])) == \
           {'interaction': 5, 'item': 1193, 'user': 1, 'timestamp': 978300760}


def test_get_test_dataset_6():
    ret = get_test_dataset('ml-10m')
    assert (len(ret), len(ret.columns)) == (698780, 5)
    assert next(ret.values(columns=['interaction', 'item', 'user', 'timestamp'])) == \
           {'user': 1, 'item': 122, 'interaction': 5.0, 'timestamp': 838985046.0}


def test_get_test_dataset_7():
    ret = get_test_dataset('ml-20m')
    assert (len(ret), len(ret.columns)) == (1384930, 5)
    assert next(ret.values(columns=['interaction', 'item', 'user', 'timestamp'])) == \
           {'user': 1, 'item': 1208, 'interaction': 3.5, 'timestamp': 1112484815.0}


""" get_train_dataset """


def test_get_train_dataset_0():
    try:
        get_train_dataset('')
    except FileNotFoundError as e:
        assert str(e) == '"" is not a valid dataset. Supported datasets: ml-100k, ml-1m, ml-10m, ml-20m, bx.'


def test_get_train_dataset_1():
    ret = get_train_dataset('ml-100k')
    assert (len(ret), len(ret.columns)) == (90570, 5)
    assert next(ret.values(columns=['interaction', 'item', 'user', 'timestamp'])) == \
           {'user': 1, 'item': 1, 'interaction': 5, 'timestamp': 874965758}


def test_get_train_dataset_2():
    ret = get_train_dataset('ml-100k')
    assert isinstance(ret, MemoryInteractionDataset)


def test_get_train_dataset_3():
    ret = get_train_dataset('ml-100k', force_out_of_memory=True)
    assert isinstance(ret, DatabaseInteractionDataset)


def test_get_train_dataset_4():
    ret = get_train_dataset('bx')
    assert (len(ret), len(ret.columns)) == (845183, 4)
    assert next(ret.values(columns=['interaction', 'item', 'user'])) == \
           {'user': 276762, 'item': '0380711524', 'interaction': 5}


def test_get_train_dataset_5():
    ret = get_train_dataset('ml-1m')
    assert (len(ret), len(ret.columns)) == (939809, 5)
    assert next(ret.values(columns=['interaction', 'item', 'user', 'timestamp'])) == \
           {'user': 1, 'item': 661, 'interaction': 3, 'timestamp': 978302109}


def test_get_train_dataset_6():
    ret = get_train_dataset('ml-10m')
    assert (len(ret), len(ret.columns)) == (9301274, 5)
    assert next(ret.values(columns=['interaction', 'item', 'user', 'timestamp'])) == \
           {'user': 1.0, 'item': 231.0, 'interaction': 5.0, 'timestamp': 838983392.0}


def test_get_train_dataset_7():
    ret = get_train_dataset('ml-20m')
    assert (len(ret), len(ret.columns)) == (18615333, 5)
    assert next(ret.values(columns=['interaction', 'item', 'user', 'timestamp'])) == \
           {'user': 1.0, 'item': 2.0, 'interaction': 3.5, 'timestamp': 1112486027.0, 'rid': 0}
