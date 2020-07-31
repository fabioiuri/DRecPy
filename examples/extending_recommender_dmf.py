from DRecPy.Recommender import DMF
import tensorflow as tf


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

