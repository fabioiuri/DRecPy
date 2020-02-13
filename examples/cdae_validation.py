from DRecPy.Recommender import CDAE
from DRecPy.Dataset import get_train_dataset
from DRecPy.Dataset import get_test_dataset
from DRecPy.Evaluation import ranking_evaluation
from DRecPy.Evaluation import predictive_evaluation
from DRecPy.Evaluation import leave_k_out
import time


ds_train = get_train_dataset('ml-100k')
ds_test = get_test_dataset('ml-100k')
ds_train, ds_val = leave_k_out(ds_train, k=5, min_user_interactions=10, seed=10)


def epoch_callback_fn(model):
    train_res = {'train_' + k: v for k, v in
                 ranking_evaluation(model, pos_interactions=1, neg_interactions=19, n_test_users=50, verbose=False, k=10).items()
                 if 'HR' in k}
    val_res = {'val_' + k: v for k, v in
               ranking_evaluation(model, ds_val, pos_interactions=1, neg_interactions=19, n_test_users=50, verbose=False, k=10).items()
               if 'HR' in k}
    return dict(train_res, **val_res)


start_train = time.time()
cdae = CDAE(min_interaction=0, seed=10)
cdae.fit(ds_train, epochs=20, learning_rate=0.0001, reg_rate=0.001,
         epoch_callback_fn=epoch_callback_fn, epoch_callback_freq=1)
print("Training took", time.time() - start_train)

print(ranking_evaluation(cdae, ds_test, n_test_users=100, seed=10))
print(predictive_evaluation(cdae, ds_test, skip_errors=True))
