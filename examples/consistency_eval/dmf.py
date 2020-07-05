from DRecPy.Recommender import DMF
from DRecPy.Dataset import get_full_dataset
from DRecPy.Evaluation.Splits import leave_k_out
from DRecPy.Evaluation.Processes import ranking_evaluation
from DRecPy.Evaluation.Metrics import ndcg
from DRecPy.Evaluation.Metrics import hit_ratio
import time

ds = get_full_dataset('ml-100k')
ds_train, ds_test = leave_k_out(ds, k=1, last_timestamps=True, seed=10)

ds_train_bin = ds_train.copy()
ds_train_bin.apply('interaction', lambda x: 1)
ds_test_bin = ds_test.copy()
ds_test_bin.apply('interaction', lambda x: 1)

for nce in [True, False]:
    print('NCE =', nce)
    start_train = time.time()
    dmf = DMF(use_nce=nce, user_factors=[128, 64], item_factors=[128, 64], seed=10)
    dmf.fit(ds_train if nce else ds_train_bin, epochs=50, batch_size=256, learning_rate=0.001, reg_rate=0.0001, neg_ratio=5)
    print("Training took", time.time() - start_train)

    print(ranking_evaluation(dmf, ds_test if nce else ds_test_bin, n_pos_interactions=1, n_neg_interactions=100, generate_negative_pairs=True,
                             novelty=True, k=list(range(1, 11)), metrics={'HR': (hit_ratio, {}), 'NDCG': (ndcg, {})},
                             seed=10))
