"""Implementation of the DMF model (Deep Matrix Factorization)
Paper: Xue, Hong-Jian, et al. "Deep Matrix Factorization Models for Recommender Systems." IJCAI. 2017."""
from DRecPy.Recommender import RecommenderABC
import tensorflow as tf
from DRecPy.Sampler import PointSampler


class DMF(RecommenderABC):
    """Deep Matrix Factorization (DMF) recommender model.

    Args:
        user_factors: A list containing the number of hidden neurons in each layer of the user NN. Default: [64, 32].
        item_factors: A list containing the number of hidden neurons in each layer of the item NN. Default: [64, 32].
        use_nce: A boolean indicating whether to use the normalized cross-entropy described in the paper as the
            loss function or the regular cross-entropy. Default: true.
        l2_norm_vectors: A boolean indicating if user and item interaction vectors should be l2 normalized before
            being used as input for their respective NNs or not. Default: true.

    For more arguments, refer to the base class: :obj:`DRecPy.Recommender.RecommenderABC`.
    """

    def __init__(self, user_factors=None, item_factors=None, use_nce=True, l2_norm_vectors=True, **kwds):
        super(DMF, self).__init__(**kwds)

        self.user_factors = user_factors
        if self.user_factors is None:
            self.user_factors = [64, 32]

        assert type(self.user_factors) is list, 'The "user_factors" argument must be of type list (ex: [64, 32]).'
        assert len(self.user_factors) > 0, 'The "user_factors" argument must have at least 1 element.'

        self.item_factors = item_factors
        if self.item_factors is None:
            self.item_factors = [64, 32]

        assert type(self.item_factors) is list, 'The "item_factors" argument must be of type list (ex: [64, 32]).'
        assert len(self.item_factors) > 0, 'The "item_factors" argument must have at least 1 element.'

        assert self.user_factors[-1] == self.item_factors[-1], f'The last user and item factors dimension must ' \
                                                               f'be equal ({self.user_factors[-1]} != {self.item_factors[-1]})'

        self.use_nce = use_nce
        self.l2_norm_vectors = l2_norm_vectors
        self._loss = tf.losses.BinaryCrossentropy()

    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        l2_reg = tf.keras.regularizers.l2(reg_rate)
        self.user_nn = tf.keras.Sequential()
        self.user_nn.add(tf.keras.layers.Dense(self.user_factors[0], activation=tf.nn.relu,
                                               input_shape=(self.n_items,), kernel_regularizer=l2_reg, autocast=False))
        for user_factor in self.user_factors[1:]:
            self.user_nn.add(tf.keras.layers.Dense(user_factor, activation=tf.nn.relu, kernel_regularizer=l2_reg))

        self.item_nn = tf.keras.Sequential()
        self.item_nn.add(tf.keras.layers.Dense(self.item_factors[0], activation=tf.nn.relu,
                                               input_shape=(self.n_users,), kernel_regularizer=l2_reg, autocast=False))
        for item_factor in self.item_factors[1:]:
            self.item_nn.add(tf.keras.layers.Dense(item_factor, activation=tf.nn.relu, kernel_regularizer=l2_reg))

        self._register_trainable(self.user_nn.trainable_variables)
        self._register_trainable(self.item_nn.trainable_variables)

        self._sampler = PointSampler(self.interaction_dataset, neg_ratio, self.interaction_threshold, self.seed)

    def _sample_batch(self, batch_size, **kwds):
        user_vecs, item_vecs, desired_values = [], [], []
        for uid, iid, interaction_value in self._sampler.sample(batch_size):
            user_vecs.append(self.interaction_dataset.select_user_interaction_vec(uid).toarray().ravel())
            item_vecs.append(self.interaction_dataset.select_item_interaction_vec(iid).toarray().ravel())
            desired_values.append(self._standardize_value(interaction_value) if self.use_nce else interaction_value)

        user_tensors = tf.convert_to_tensor(user_vecs, dtype=tf.float32)
        item_tensors = tf.convert_to_tensor(item_vecs, dtype=tf.float32)

        if self.l2_norm_vectors:
            user_tensors = tf.nn.l2_normalize(user_tensors, axis=1)
            item_tensors = tf.nn.l2_normalize(item_tensors, axis=1)

        return {'user_tensors': user_tensors, 'item_tensors': item_tensors, 'desired_values': desired_values}

    def _predict_batch(self, batch_samples, **kwds):
        user_reps = self.user_nn(batch_samples['user_tensors'])
        item_reps = self.item_nn(batch_samples['item_tensors'])

        norm_user_reps = tf.nn.l2_normalize(user_reps, axis=1)
        norm_item_reps = tf.nn.l2_normalize(item_reps, axis=1)

        predictions = tf.maximum(1e-6, tf.reduce_sum(tf.multiply(norm_user_reps, norm_item_reps), axis=1))
        return predictions, batch_samples['desired_values']

    def _compute_batch_loss(self, predictions, desired_values, **kwds):
        return self._loss(desired_values, predictions) + \
               tf.math.add_n(self.user_nn.losses) + tf.math.add_n(self.item_nn.losses)

    def _predict(self, uid, iid, **kwds):
        user_vec = self.interaction_dataset.select_user_interaction_vec(uid).toarray().ravel()
        item_vec = self.interaction_dataset.select_item_interaction_vec(iid).toarray().ravel()

        user_tensor = tf.convert_to_tensor([user_vec], dtype=tf.float32)
        item_tensor = tf.convert_to_tensor([item_vec], dtype=tf.float32)

        if self.l2_norm_vectors:
            user_tensor = tf.nn.l2_normalize(user_tensor)
            item_tensor = tf.nn.l2_normalize(item_tensor)

        user_rep = self.user_nn(user_tensor)
        item_rep = self.item_nn(item_tensor)

        norm_user_rep = tf.nn.l2_normalize(user_rep)
        norm_item_rep = tf.nn.l2_normalize(item_rep)

        pred = tf.reduce_sum(tf.multiply(norm_user_rep, norm_item_rep))
        return self._rescale_value(pred)
