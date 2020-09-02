from DRecPy.Recommender.Baseline import ItemKNN
from DRecPy.Dataset import get_train_dataset
from DRecPy.Dataset import get_test_dataset
from DRecPy.Evaluation.Processes import predictive_evaluation
import time

ds_train = get_train_dataset('ml-100k')
ds_test = get_test_dataset('ml-100k')

start_train = time.time()
item_cf = ItemKNN(k=15, m=1, shrinkage=100, sim_metric='adjusted_cosine', verbose=True)
item_cf.fit(ds_train)
print("Training took", time.time() - start_train)

start_evaluation = time.time()
print(predictive_evaluation(item_cf, ds_test))
print("Evaluation took", time.time() - start_evaluation)

