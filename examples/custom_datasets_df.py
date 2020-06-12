from DRecPy.Dataset import InteractionDataset
import pandas as pd
from os import remove

# create file with sample dataset
with open('tmp.csv', 'w') as f:
    f.write('users,items,interactions\n')
    f.write('"john","ps4",4.5\n')
    f.write('"patrick","xbox",4.1\n')
    f.write('"anna","brush",3.6\n')
    f.write('"david","tv",2.0\n')

# load dataset into memory
df = pd.read_csv('tmp.csv')
ds_memory = InteractionDataset.read_df(df, user_label='users', item_label='items', interaction_label='interactions')
print('all values:', ds_memory.values_list())

remove('tmp.csv')  # delete previously created sample dataset file
