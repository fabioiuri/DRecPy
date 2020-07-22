from DRecPy.Recommender import RecommenderABC
from DRecPy.Sampler import PointSampler
import tensorflow as tf
from DRecPy.Dataset import get_train_dataset


class TestRecommender(RecommenderABC):

    def __init__(self, **kwds):
        super(TestRecommender, self).__init__(**kwds)

    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        # used to declare variables and the neural network structure of the model, as well as register trainable vars
        self._log(f'doing pre-fit with learning_rate={learning_rate}, neg_ratio={neg_ratio}, reg_rate={reg_rate}')
        self._weights = tf.Variable([[0.5], [0.5]])
        self._register_trainable(self._weights)
        self._loss = tf.losses.BinaryCrossentropy()
        self._sampler = PointSampler(self.interaction_dataset, neg_ratio=neg_ratio)

    def _sample_batch(self, batch_size, **kwds):
        self._log(f'doing _sample_batch {batch_size}')
        return self._sampler.sample(batch_size)

    def _predict_batch(self, batch_samples, **kwds):
        # must return predictions from which gradients can be computed in order to update the registered trainable vars
        # and the desired values, so that we're able to compute the batch loss
        self._log(f'doing _predict_batch {batch_samples}')
        predictions = [self._predict(u, i) for u, i, _ in batch_samples]
        desired_values = [y for _, _, y in batch_samples]
        self._log(f'predictions = {predictions}, desired_values = {desired_values}')
        return predictions, desired_values

    def _compute_batch_loss(self, predictions, desired_values, **kwds):
        # receives the predictions and desired values computed during the _predict_batch, and should apply a loss
        # function from which gradients can then be computed
        self._log(f'doing _compute_batch_loss: predictions={predictions}, desired_values={desired_values}')
        return self._loss(desired_values, predictions)

    def _predict(self, uid, iid, **kwds):
        # predict for a (user, item) pair
        return tf.sigmoid(tf.matmul(tf.convert_to_tensor([[uid, iid]], dtype=tf.float32), self._weights))


ds_train = get_train_dataset('ml-100k', verbose=False)

print('TestRecommender')
recommender = TestRecommender(verbose=True)
recommender.fit(ds_train, epochs=2, batch_size=10)
print(recommender.predict(1, 1))
