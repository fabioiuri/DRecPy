import os
from zipfile import ZipFile
import requests
from DRecPy.Dataset import InteractionDataset
from DRecPy.Evaluation.Splits import leave_k_out
from .file_utils import get_dataset_path
from .file_utils import is_stored


class DatasetReadConfig:
    def __init__(self, url, full_file, columns, delimiter, encoding='utf8',
                 train_file=None, test_file=None, unzip_folder=None, has_header=False):
        self.url = url
        self.train_file = train_file
        self.test_file = test_file
        self.full_file = full_file
        self.columns = columns
        self.delimiter = delimiter
        self.encoding = encoding
        self.unzip_folder = unzip_folder
        self.has_header = has_header


DATASETS = {
    'ml-100k': DatasetReadConfig(url='http://files.grouplens.org/datasets/movielens/ml-100k.zip',
                                 train_file='ua.base',
                                 test_file='ua.test',
                                 full_file='u.data',
                                 columns=['user', 'item', 'interaction', 'timestamp'],
                                 delimiter='\t',
                                 unzip_folder='ml-100k'),
    'ml-1m': DatasetReadConfig(url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
                               full_file='ratings.dat',
                               columns=['user', 'item', 'interaction', 'timestamp'],
                               delimiter='::',
                               unzip_folder='ml-1m'),
    'ml-10m': DatasetReadConfig(url='http://files.grouplens.org/datasets/movielens/ml-10m.zip',
                                full_file='ratings.dat',
                                columns=['user', 'item', 'interaction', 'timestamp'],
                                delimiter='::',
                                unzip_folder='ml-10M100K'),
    'ml-20m': DatasetReadConfig(url='http://files.grouplens.org/datasets/movielens/ml-20m.zip',
                                full_file='ratings.csv',
                                columns=['user', 'item', 'interaction', 'timestamp'],
                                delimiter=',',
                                unzip_folder='ml-20m',
                                has_header=True),
    'bx': DatasetReadConfig(url='http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip',
                            full_file='BX-Book-Ratings.csv',
                            columns=['user', 'item', 'interaction'],
                            delimiter=';',
                            encoding='latin1',
                            has_header=True)
}


def download_dataset(ds_name):
    """Download the dataset with name passed as argument."""
    ds_options = DATASETS[ds_name]
    print('> Downloading from', ds_options.url)
    data = requests.get(ds_options.url).content

    dataset_path = get_dataset_path(ds_name)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    print('> Extracting files from zip')
    tmp_path = os.path.join(dataset_path, 'tmp.zip')
    with open(tmp_path, "wb") as f:
        f.write(data)

    with ZipFile(tmp_path, 'r') as tmp_zip:
        tmp_zip.extractall(dataset_path)

    # if dataset comes in folder, move it to dataset root folder
    if ds_options.unzip_folder is not None:
        src = os.path.join(dataset_path, ds_options.unzip_folder)
        for f in os.listdir(src):
            _from = os.path.join(src, f)
            _to = os.path.join(dataset_path, f)
            os.rename(_from, _to)
        os.removedirs(src)

    os.remove(tmp_path)


def get_dataset(ds_name, path, is_generated=False, force_out_of_memory=False, verbose=True):
    """Returns an InteractionDataset containing the data present in the path argument, and uses the
    settings defined for the dataset specified in the ds_name argument. Downloads the dataset
    if is not already stored."""
    ds_options = DATASETS[ds_name]

    if not is_stored(ds_name):
        download_dataset(ds_name)

    if is_generated:
        return InteractionDataset(path, delimiter=',', columns=ds_options.columns, encoding=ds_options.encoding,
                                  in_memory=not force_out_of_memory, verbose=verbose)
    else:
        return InteractionDataset(path, delimiter=ds_options.delimiter, columns=ds_options.columns,
                                  encoding=ds_options.encoding, has_header=ds_options.has_header,
                                  in_memory=not force_out_of_memory, verbose=verbose)


def get_train_dataset(ds_name, force_out_of_memory=False, verbose=True):
    """Gets a train dataset. If the named dataset does not have a specific train file
    (example: BX dataset), a train InteractionDataset will be created using leave_k_out() from the Evaluation module
    on the full dataset. The split is deterministic (i.e. has a defined seed value).
    Might download the dataset if it hasn't been downloaded before.

    Args:
        ds_name: A string with the name of the requested dataset.
            This name should be present in the list returned by available_datasets(),
            otherwise an error will be thrown.
        force_out_of_memory: A boolean indicating whether to force dataset loading to out of memory. Default: False.
        verbose: A boolean indicating whether to log info messages or not. Default: True.

    Returns:
        A InteractionDataset containing the train dataset.
    """
    if ds_name not in DATASETS:
        raise FileNotFoundError(f'"{ds_name}" is not a valid dataset. Supported datasets: {", ".join(available_datasets())}.')

    ds_options = DATASETS[ds_name]
    if ds_options.train_file is None:
        generated_path = os.path.join(get_dataset_path(ds_name), ds_name + '_train.gen')
        if os.path.exists(generated_path):  # might have been generated already
            return get_dataset(ds_name, generated_path, is_generated=True, force_out_of_memory=force_out_of_memory,
                               verbose=verbose)

        # need to generate it now
        path = os.path.join(get_dataset_path(ds_name), ds_options.full_file)
        full_ds = get_dataset(ds_name, path, force_out_of_memory=force_out_of_memory, verbose=verbose)
        train_ds, test_ds = leave_k_out(full_ds, k=10, min_user_interactions=10, seed=10)

        # store generated datasets for future calls
        train_ds.save(os.path.join(get_dataset_path(ds_name), ds_name + '_train.gen'))
        test_ds.save(os.path.join(get_dataset_path(ds_name), ds_name + '_test.gen'))
        return train_ds

    path = os.path.join(get_dataset_path(ds_name), ds_options.train_file)
    return get_dataset(ds_name, path, force_out_of_memory=force_out_of_memory, verbose=verbose)


def get_test_dataset(ds_name, force_out_of_memory=False, verbose=True):
    """Gets a test dataset. If the named dataset does not have a specific test file
    (example: BX dataset), a test InteractionDataset will be created using leave_k_out() from the Evaluation module
    on the full dataset. The split is deterministic (i.e. has a defined seed value).
    Might download the dataset if it hasn't been downloaded before.

    Args:
        ds_name: A string with the name of the requested dataset.
            This name should be present in the list returned by available_datasets(),
            otherwise an error will be thrown.
        force_out_of_memory: A boolean indicating whether to force dataset loading to out of memory. Default: False.
        verbose: A boolean indicating whether to log info messages or not. Default: True.

    Returns:
        A InteractionDataset containing the test dataset.
    """
    if ds_name not in DATASETS:
        raise FileNotFoundError(f'"{ds_name}" is not a valid dataset. Supported datasets: {", ".join(available_datasets())}.')

    ds_options = DATASETS[ds_name]
    if ds_options.test_file is None:
        generated_path = os.path.join(get_dataset_path(ds_name), ds_name + '_test.gen')
        if os.path.exists(generated_path):  # might have been generated already
            return get_dataset(ds_name, generated_path, is_generated=True, force_out_of_memory=force_out_of_memory,
                               verbose=verbose)

        # need to generate it now
        path = os.path.join(get_dataset_path(ds_name), ds_options.full_file)
        full_ds = get_dataset(ds_name, path, force_out_of_memory=force_out_of_memory, verbose=verbose)
        train_ds, test_ds = leave_k_out(full_ds, k=10, min_user_interactions=10, seed=10)

        # store generated datasets for future calls
        train_ds.save(os.path.join(get_dataset_path(ds_name), ds_name + '_train.gen'))
        test_ds.save(os.path.join(get_dataset_path(ds_name), ds_name + '_test.gen'))
        return test_ds

    path = os.path.join(get_dataset_path(ds_name), ds_options.test_file)
    return get_dataset(ds_name, path, force_out_of_memory=force_out_of_memory, verbose=verbose)


def get_full_dataset(ds_name, force_out_of_memory=False, verbose=True):
    """Gets a full dataset. Might download the dataset if it hasn't been downloaded before.

    Args:
        ds_name: A string with the name of the requested dataset.
            This name should be present in the list returned by available_datasets(),
            otherwise an error will be thrown.
        force_out_of_memory: A boolean indicating whether to force dataset loading to out of memory. Default: False.
        verbose: A boolean indicating whether to log info messages or not. Default: True.

    Returns:
        A InteractionDataset containing the dataset.
    """
    if ds_name not in DATASETS:
        raise FileNotFoundError(f'"{ds_name}" is not a valid dataset. Supported datasets: {", ".join(available_datasets())}.')

    path = os.path.join(get_dataset_path(ds_name), DATASETS[ds_name].full_file)
    return get_dataset(ds_name, path, force_out_of_memory=force_out_of_memory, verbose=verbose)


def available_datasets():
    """Returns a list of the datasets available to download."""
    return list(DATASETS.keys())
