"""Implementation of the CDAE model (Collaborative Denoising Auto-Encoder).
Paper: Wu, Yao, et al. "Collaborative denoising auto-encoders for top-n recommender systems." Proceedings
of the Ninth ACM International Conference on Web Search and Data Mining. ACM, 2016.

Note: gradients are evaluated for all output units (the output unit selection step discussed in the paper is not done).
"""
from DRecPy.Recommender import RecommenderABC
import tensorflow as tf
from heapq import nlargest


class CDAE(RecommenderABC):
    """Collaborative Denoising Auto-Encoder (CDAE) recommender model.

    Args:
        hidden_factors: An integer defining the number of units for the hidden layer.
        corruption_level: A decimal value representing the level of corruption to apply to the
            given interactions / ratings during training.
        loss: The loss function used to optimize the model. Supported: mse, bce.

    For more arguments, refer to the base class: :obj:`DRecPy.Recommender.RecommenderABC`.
    """
    def __init__(self, hidden_factors=50, corruption_level=0.2, loss='mse', **kwds):
        super(CDAE, self).__init__(**kwds)

        self.hidden_factors = hidden_factors
        self.corruption_level = corruption_level
        if loss == 'mse':  self.loss = tf.losses.MeanSquaredError()
        elif loss == 'bce': self.loss = tf.losses.BinaryCrossentropy()
        else: raise Exception(f'Loss function "{loss}" is not supported. Supported losses: "mse", "bce".')

    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        weight_initializer = tf.initializers.GlorotUniform()
        self.W = tf.Variable(weight_initializer(shape=[self.n_items, self.hidden_factors]))
        self.W_ = tf.Variable(weight_initializer(shape=[self.hidden_factors, self.n_items]))
        self.V = tf.Variable(weight_initializer(shape=[self.n_users, self.hidden_factors]))

        self.b = tf.Variable(weight_initializer(shape=[self.hidden_factors]))
        self.b_ = tf.Variable(weight_initializer(shape=[self.n_items]))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.reg = lambda W, W_, V, b, b_: \
            (tf.nn.l2_loss(W) + tf.nn.l2_loss(W_) + tf.nn.l2_loss(V) + tf.nn.l2_loss(b) + tf.nn.l2_loss(b_)) \
            * reg_rate / 2

    def _do_batch(self, **kwds):
        sampled_uid = self._rng.randint(0, self.n_users-1)
        with tf.GradientTape() as tape:
            tape.watch(self.W), tape.watch(self.W_), tape.watch(self.V), tape.watch(self.b), tape.watch(self.b_)
            real_preferences, predictions = self._reconstruct(sampled_uid, corrupt=True)
            loss = self.loss(real_preferences, predictions) + self.reg(self.W, self.W_, self.V, self.b, self.b_)

        grads = tape.gradient(loss, [self.W, self.W_, self.V, self.b, self.b_])
        self.optimizer.apply_gradients(zip(grads, [self.W, self.W_, self.V, self.b, self.b_]))

        return loss

    def _reconstruct(self, uid, corrupt=False):
        """Gathers the user embedding vector, its interaction vector (and corrupts it when corrupt=True,
        which happens during training only), and computes the hidden layer values and finally the output
        layer values."""
        user_embedding = tf.nn.embedding_lookup(self.V, uid)
        user_interaction_vec = self.interaction_dataset.select_user_interaction_vec(uid).toarray().ravel()

        if corrupt:
            user_corrupted_interaction_vec = [0. if interaction == 0 or self._rng.uniform(0, 1) < self.corruption_level
                                              else self._standardize_value(interaction / (1 - self.corruption_level))
                                              for interaction in user_interaction_vec]
            hidden_layer = tf.sigmoid(
                tf.matmul(tf.convert_to_tensor([user_corrupted_interaction_vec], dtype=tf.float32), self.W) +
                user_embedding + self.b)  # I x K (k = hidden factors)
        else:
            user_interaction_vec = [self._standardize_value(i) for i in user_interaction_vec]
            hidden_layer = tf.sigmoid(
                tf.matmul(tf.convert_to_tensor([user_interaction_vec], dtype=tf.float32), self.W) +
                user_embedding + self.b)  # I x K (k = hidden factors)

        output_layer = tf.sigmoid(tf.matmul(hidden_layer, self.W_) + self.b_)  # K x I
        return user_interaction_vec, output_layer

    def _predict(self, uid, iid, **kwds):
        if uid is None or iid is None: return None

        _, predictions = self._reconstruct(uid)
        predictions = predictions.numpy().ravel()
        return predictions[iid]

    def _rank(self, uid, iids, n, novelty):
        _, predictions = self._reconstruct(uid)
        predictions = predictions.numpy().ravel()

        if novelty:
            user_ds = self.interaction_dataset.select(f'uid == {uid}')
            iids = set([iid for iid in iids if not user_ds.exists(f'iid == {iid}')])
        else:
            iids = set(iids)

        pred_list = [(prediction, iid) for prediction, iid in zip(predictions, range(self.n_items)) if iid in iids]
        return nlargest(n, pred_list)
