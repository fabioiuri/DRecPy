"""Implementation of the CDAE model (Collaborative Denoising Auto-Encoder).
Paper: Wu, Yao, et al. "Collaborative denoising auto-encoders for top-n recommender systems." Proceedings
of the Ninth ACM International Conference on Web Search and Data Mining. ACM, 2016.

Note: gradients are evaluated for all output units (the output unit selection step discussed in the paper is not done).
"""
from DRecPy.Recommender import RecommenderABC
import tensorflow as tf
from heapq import nlargest
from DRecPy.Sampler import PointSampler


class CDAE(RecommenderABC):
    """Collaborative Denoising Auto-Encoder (CDAE) recommender model.

    Args:
        hidden_factors: An integer defining the number of units for the hidden layer.
        corruption_level: A decimal value representing the level of corruption to apply to the
            given interactions / ratings during training.
        loss: A string that represents the loss function used to optimize the model.
            Supported: mse, bce. Default: bce.

    For more arguments, refer to the base class: :obj:`DRecPy.Recommender.RecommenderABC`.
    """
    def __init__(self, hidden_factors=50, corruption_level=0.2, loss='bce', **kwds):
        super(CDAE, self).__init__(**kwds)

        self.hidden_factors = hidden_factors
        self.corruption_level = corruption_level
        if loss == 'mse':  self._loss = tf.losses.MeanSquaredError()
        elif loss == 'bce': self._loss = tf.losses.BinaryCrossentropy()
        else: raise Exception(f'Loss function "{loss}" is not supported. Supported losses: "mse", "bce".')

    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        weight_initializer = tf.initializers.GlorotUniform()
        self.W = tf.Variable(weight_initializer(shape=[self.n_items, self.hidden_factors]))
        self.W_ = tf.Variable(weight_initializer(shape=[self.hidden_factors, self.n_items]))
        self.V = tf.Variable(weight_initializer(shape=[self.n_users, self.hidden_factors]))

        self.b = tf.Variable(weight_initializer(shape=[self.hidden_factors]))
        self.b_ = tf.Variable(weight_initializer(shape=[self.n_items]))

        self._register_trainables([self.W, self.W_, self.V, self.b, self.b_])

        self._sampler = PointSampler(self.interaction_dataset, neg_ratio, self.interaction_threshold, self.seed)

    def _sample_batch(self, batch_size, **kwds):
        return self._sampler.sample(batch_size)

    def _predict_batch(self, batch_samples, **kwds):
        predictions, desired_values = [], []
        for uid, _, _ in batch_samples:
            prediction_vector, desired_vector = self._reconstruct_for_training(uid)
            predictions.append(prediction_vector)
            desired_values.append(desired_vector)

        return predictions, desired_values

    def _reconstruct_for_training(self, uid):
        user_embedding = tf.nn.embedding_lookup(self.V, uid)
        desired_vector = [1 if i >= self.interaction_threshold else 0 for i in
                          self.interaction_dataset.select_user_interaction_vec(uid).toarray().ravel()]
        corrupted_vector = [0. if self._rng.uniform(0, 1) < self.corruption_level
                            else i / (1 - self.corruption_level) for i in desired_vector]
        return self._reconstruct(user_embedding, corrupted_vector), desired_vector

    def _reconstruct_for_predictions(self, uid):
        user_embedding = tf.nn.embedding_lookup(self.V, uid)
        desired_vector = [1 if i >= self.interaction_threshold else 0 for i in
                          self.interaction_dataset.select_user_interaction_vec(uid).toarray().ravel()]
        return self._reconstruct(user_embedding, desired_vector).numpy().ravel()

    def _reconstruct(self, user_embedding, input_vector, **kwds):
        hidden_layer = tf.sigmoid(tf.matmul(tf.convert_to_tensor([input_vector], dtype=tf.float32), self.W) +
                                  user_embedding + self.b)  # I x K (k = hidden factors)
        return tf.sigmoid(tf.matmul(hidden_layer, self.W_) + self.b_)  # K x I

    def _compute_batch_loss(self, predictions, desired_values, **kwds):
        return self._loss(desired_values, predictions)

    def _compute_reg_loss(self, reg_rate, batch_size, **kwds):
        return sum([tf.nn.l2_loss(v) for v in [self.W, self.W_, self.V, self.b, self.b_]]) * reg_rate / (2 * batch_size)

    def _predict(self, uid, iid=None, **kwds):
        if uid is None: return None

        predictions = self._reconstruct_for_predictions(uid)
        return predictions if iid is None else predictions[iid]

    def _rank(self, uid, iids, n, novelty):
        predictions = self._predict(uid)

        if novelty:
            user_ds = self.interaction_dataset.select(f'uid == {uid}')
            if len(user_ds) < len(iids):
                iids = set(iids).difference(set(user_ds.values_list('iid', to_list=True)))
            else:
                iids = set([iid for iid in iids if not user_ds.exists(f'iid == {iid}')])
        else:
            iids = set(iids)

        pred_list = [(prediction, iid) for prediction, iid in zip(predictions, range(self.n_items)) if iid in iids]
        return nlargest(n, pred_list)
