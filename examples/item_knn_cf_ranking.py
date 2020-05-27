from DRecPy.Recommender.Baseline import ItemKNN
from DRecPy.Dataset import get_full_dataset
from DRecPy.Evaluation.Splits import leave_k_out
from DRecPy.Evaluation.Processes import ranking_evaluation
from DRecPy.Evaluation.Metrics import hit_ratio
from DRecPy.Evaluation.Metrics import ndcg
import time

ds_full = get_full_dataset('ml-100k')
ds_train, ds_test = leave_k_out(ds_full, k=1, last_timestamps=True, seed=0)

start_train = time.time()
item_cf = ItemKNN(k=5, m=1, shrinkage=50, sim_metric='adjusted_cosine', verbose=True)
item_cf.fit(ds_train)
print("Training took", time.time() - start_train)

start_evaluation = time.time()
print(ranking_evaluation(item_cf, ds_test,  n_pos_interactions=1, n_neg_interactions=100,  generate_negative_pairs=True,
                         novelty=True, k=list(range(1, 11)), metrics={'HR': (hit_ratio, {}), 'NDCG': (ndcg, {})},
                         seed=10))
print("Evaluation took", time.time() - start_evaluation)
