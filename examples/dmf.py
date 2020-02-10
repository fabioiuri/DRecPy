from DRecPy.Recommender import DMF
from DRecPy.Dataset import get_train_dataset
from DRecPy.Dataset import get_test_dataset
from DRecPy.Evaluation import predictive_evaluation
from DRecPy.Evaluation import ranking_evaluation
import time

ds_train = get_train_dataset('ml-100k')
ds_test = get_test_dataset('ml-100k')

start_train = time.time()
dmf = DMF(batch_size=256, user_factors=[128, 64], item_factors=[128, 64],  min_interaction=0, seed=10)
dmf.fit(ds_train, epochs=75, learning_rate=0.0001, neg_ratio=5)
print("Training took", time.time() - start_train)

print(ranking_evaluation(dmf, ds_test, n_test_users=100, seed=10))
print(predictive_evaluation(dmf, ds_test, skip_errors=True))