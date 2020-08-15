import os


def data_path():
    """Auxiliary method to create (if needed) and to return the package data path."""
    path = os.environ.get('DATA_FOLDER', os.path.join(os.path.expanduser('~'), '.DRecPy_data'))
    if not os.path.exists(path):
        print('> Creating data path at', path)
        os.makedirs(path)
    return path


def get_dataset_path(ds_name):
    """Auxiliary method to build the dataset path given the dataset name."""
    return os.path.join(data_path(), ds_name)


def is_stored(ds_name):
    """Verifies if the dataset with name passed as argument is already stored."""
    ds_path = get_dataset_path(ds_name)
    return os.path.exists(ds_path)


def register_temp_file(file_path):
    try:
        with open(os.path.join(data_path(), 'tmp_files.txt'), 'a') as f:
            f.write(file_path + '\n')
    except NameError: pass  # possible errors with calling 'open' on __del__
    except FileNotFoundError: pass


def unregister_temp_file(file_path):
    empty = True

    try:
        with open(os.path.join(data_path(), 'tmp_files.txt'), 'r') as f:
            paths = f.readlines()
        with open(os.path.join(data_path(), 'tmp_files.txt'), 'w') as f:
            for path in paths:
                if path.strip("\n") != file_path:
                    f.write(path)
                    empty = False
    except NameError: pass
    except FileNotFoundError: pass

    if empty:
        os.remove(os.path.join(data_path(), 'tmp_files.txt'))


def delete_temp_files():
    try:
        with open(os.path.join(data_path(), 'tmp_files.txt'), 'r') as f:
            for path in f:
                try:
                    os.remove(path.strip())
                except: pass
        os.remove(os.path.join(data_path(), 'tmp_files.txt'))
    except NameError: pass
    except FileNotFoundError: pass