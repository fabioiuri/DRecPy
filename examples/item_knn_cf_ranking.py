from DRecPy.Recommender.Baseline import ItemKNN
from DRecPy.Dataset import get_full_dataset
from DRecPy.Evaluation import leave_k_out
from DRecPy.Evaluation import ranking_evaluation
import time

ds_full = get_full_dataset('ml-1m')
ds_full.apply('interaction', lambda x: 1 if x > 0 else 0)

ds_train, ds_test = leave_k_out(ds_full, seed=10)

start_train = time.time()
item_cf = ItemKNN(k=12, m=1, shrinkage=50, sim_metric='cosine', verbose=True)
item_cf.fit(ds_train)
print("Training took", time.time() - start_train)

start_evaluation = time.time()
print(ranking_evaluation(item_cf, ds_test, n_test_users=100, seed=10))
print("Evaluation took", time.time() - start_evaluation)
