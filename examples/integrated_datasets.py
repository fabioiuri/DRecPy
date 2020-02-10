from DRecPy.Dataset import get_train_dataset
from DRecPy.Dataset import get_test_dataset
from DRecPy.Dataset import get_full_dataset
from DRecPy.Dataset import available_datasets

print('Available datasets', available_datasets())

# Reading the ml-100k full dataset and prebuilt train and test datasets.
print('ml-100k full dataset', get_full_dataset('ml-100k'))
print('ml-100k train dataset', get_train_dataset('ml-100k'))
print('ml-100k test dataset', get_test_dataset('ml-100k'))

# Reading the ml-1m full dataset and generated train and test datasets using out of memory storage.
print('ml-1m full dataset', get_full_dataset('ml-1m', force_out_of_memory=True))
print('ml-1m train dataset', get_train_dataset('ml-1m', force_out_of_memory=True))
print('ml-1m test dataset', get_test_dataset('ml-1m', force_out_of_memory=True))

# Showcase some dataset operations
ds_ml = get_full_dataset('ml-100k')
print('Minimum rating value:', ds_ml.min('interaction'))
print('Unique rating values:', ds_ml.unique('interaction').values_list())

ds_ml.apply('interaction', lambda x: x / ds_ml.max('interaction'))  # standardize the rating value
print('New values', ds_ml.values_list()[:5])
