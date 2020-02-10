from DRecPy.Dataset import InteractionDataset
from os import remove

# create file with sample dataset
with open('tmp.csv', 'w') as f:
    f.write('"john","ps4",4.5\n')
    f.write('"patrick","xbox",4.1\n')
    f.write('"anna","brush",3.6\n')
    f.write('"david","tv",2.0\n')

# load dataset into memory
ds_memory = InteractionDataset('tmp.csv', columns=['user', 'item', 'interaction'])
print('all values:', ds_memory.values_list())
print('filtered values:', ds_memory.select('interaction > 3.5').values_list())
ds_memory_scaled = ds_memory.__copy__()
ds_memory_scaled.apply('interaction', lambda x: x / ds_memory.max('interaction'))
print('all values scaled:', ds_memory_scaled.values_list())

# load dataset out of memory
ds_out_of_memory = InteractionDataset('tmp.csv', columns=['user', 'item', 'interaction'], in_memory=False)
print('all values:', ds_out_of_memory.values_list())
print('filtered values:', ds_out_of_memory.select('interaction > 3.5').values_list())

remove('tmp.csv')  # delete previously created sample dataset file
