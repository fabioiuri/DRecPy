from DRecPy.Recommender import DMF
from DRecPy.Evaluation.Processes import ranking_evaluation
import tensorflow as tf
from DRecPy.Dataset import get_train_dataset
from DRecPy.Dataset import get_test_dataset


class ModifiedDMF(DMF):

    def __init__(self, **kwds):
        super(ModifiedDMF, self).__init__(**kwds)

    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        super(ModifiedDMF, self)._pre_fit(learning_rate, neg_ratio, reg_rate, **kwds)
        self._extra_weight = tf.Variable([1.])
        self._register_trainable(self._extra_weight)

    def _predict_batch(self, batch_samples, **kwds):
        predictions, desired_values = super(ModifiedDMF, self)._predict_batch(batch_samples, **kwds)
        predictions = [(self._extra_weight * pred) for pred in predictions]
        return predictions, desired_values

    def _predict(self, uid, iid, **kwds):
        dmf_pred = super(ModifiedDMF, self)._predict(uid, iid)
        return self._extra_weight * dmf_pred


ds_train = get_train_dataset('ml-100k', verbose=False)
ds_test = get_test_dataset('ml-100k', verbose=False)

recommender = ModifiedDMF(use_nce=True, user_factors=[128, 64], item_factors=[128, 64], seed=10, verbose=True)
recommender.fit(ds_train, epochs=50, batch_size=256, learning_rate=0.01, reg_rate=0.0001, neg_ratio=5)

print(ranking_evaluation(recommender, ds_test, n_pos_interactions=1, n_neg_interactions=100,
                         generate_negative_pairs=True, ovelty=True, k=list(range(1, 11)), seed=10, n_test_users=100))
