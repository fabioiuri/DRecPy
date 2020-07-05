from DRecPy.Recommender import CDAE
from DRecPy.Dataset import get_train_dataset
from DRecPy.Dataset import get_test_dataset
from DRecPy.Evaluation.Processes import ranking_evaluation
from DRecPy.Evaluation.Splits import leave_k_out
from DRecPy.Evaluation.Metrics import ndcg
from DRecPy.Evaluation.Metrics import hit_ratio
import time


ds_train = get_train_dataset('ml-100k')
ds_test = get_test_dataset('ml-100k')
ds_train, ds_val = leave_k_out(ds_train, k=1, min_user_interactions=10, seed=0)


def epoch_callback_fn(model):
    return {'val_' + metric: v for metric, v in
            ranking_evaluation(model, ds_val, n_pos_interactions=1, n_neg_interactions=100,
                               generate_negative_pairs=True, k=10, verbose=False, seed=10,
                               metrics={'HR': (hit_ratio, {}), 'NDCG': (ndcg, {})}).items()}


start_train = time.time()
cdae = CDAE(hidden_factors=50, corruption_level=0.2, loss='bce', seed=10)
cdae.fit(ds_train, learning_rate=0.001, reg_rate=0.001, epochs=50, batch_size=64, neg_ratio=5,
         epoch_callback_fn=epoch_callback_fn, epoch_callback_freq=10)
print("Training took", time.time() - start_train)

print(ranking_evaluation(cdae, ds_test, k=[1, 5, 10], novelty=True, n_pos_interactions=1,
                         n_neg_interactions=100, generate_negative_pairs=True, seed=10,
                         max_concurrent_threads=4, verbose=True))