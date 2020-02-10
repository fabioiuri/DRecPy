from DRecPy.Recommender import CDAE
from DRecPy.Dataset import get_train_dataset
from DRecPy.Dataset import get_test_dataset
from DRecPy.Evaluation import ranking_evaluation
from DRecPy.Evaluation import predictive_evaluation
import time

ds_train = get_train_dataset('ml-100k')
ds_test = get_test_dataset('ml-100k')

start_train = time.time()
cdae = CDAE(min_interaction=0, seed=10)
cdae.fit(ds_train, epochs=100)
print("Training took", time.time() - start_train)

print(ranking_evaluation(cdae, ds_test, n_test_users=100, seed=10))
print(predictive_evaluation(cdae, ds_test, skip_errors=True))