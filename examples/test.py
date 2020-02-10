from DRecPy.Recommender import Test
from DRecPy.Dataset import get_train_dataset
from DRecPy.Dataset import get_test_dataset
from DRecPy.Evaluation import ranking_evaluation
import time

ds_train = get_train_dataset('ml-100k')
ds_test = get_test_dataset('ml-100k')

start_train = time.time()

test = Test(n_supported_items=5, A=256, B=128, C=64, seed=10)
test.fit(ds_train, learning_rate=0.001, epochs=50, batch_size=64)
print("Training took", time.time() - start_train)

print(ranking_evaluation(test, ds_test, n_test_users=100, seed=10))