"""Implementation of the Caser model.
Paper: Tang, Jiaxi, and Ke Wang. "Personalized top-n sequential recommendation via convolutional sequence embedding."
Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining. 2018.
"""
from DRecPy.Recommender import RecommenderABC
from DRecPy.Sampler import ListSampler
import tensorflow as tf
from heapq import nlargest


class Caser(RecommenderABC):
    """Caser recommender model.

    Args:
        L: An integer representing the sequence length. Default: 5.
        T: An integer representing the number of targets. Default: 3.
        d: An integer representing the number of latent dimensions. Default: 50.
        n_v: An integer representing the number of vertical filters. Default: 4.
        n_h: An integer representing the number of horizontal filters. Default: 16.
        act_h: The activation function used for the horizontal convolutional layer. Default: tf.nn.relu.
        act_mlp: The activation function used for the dense layer. Default: tf.nn.relu.
        dropout_rate: The dropout ratio when performing dropout between the convolutional and dense layers.
            Default: 0.5.
        sort_column: An optional string representing the name of the column used to sort the sequence records. If none
            is provided, the natural order (present in the data set) will be preserved. Default: 'timestamp'.

    For more arguments, refer to the base class: :obj:`DRecPy.Recommender.RecommenderABC`.
    """
    def __init__(self, L=5, T=3, d=50, n_v=4, n_h=16, act_h=tf.nn.relu, act_mlp=tf.nn.relu, dropout_rate=0.5,
                 sort_column='timestamp', **kwds):
        super(Caser, self).__init__(**kwds)

        self.L = L
        self.T = T
        self.d = d
        self.n_v = n_v
        self.n_h = n_h
        self.act_h = act_h
        self.act_mlp = act_mlp
        self.dropout_rate = dropout_rate
        self.sort_column = sort_column

        self._loss = tf.losses.BinaryCrossentropy()

    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        l2_reg = tf.keras.regularizers.l2(reg_rate)
        self.user_embeddings = tf.keras.layers.Embedding(self.n_users, self.d, embeddings_regularizer=l2_reg)
        self._register_trainable(self.user_embeddings)

        self.item_embeddings = tf.keras.layers.Embedding(self.n_items, self.d, embeddings_regularizer=l2_reg)
        self._register_trainable(self.item_embeddings)

        self.conv_v = tf.keras.layers.Conv1D(filters=self.n_v, kernel_size=self.L, kernel_regularizer=l2_reg)
        self._register_trainable(self.conv_v)

        self.convs_h = []
        for i in range(self.L):
            self.convs_h.append(tf.keras.layers.Conv1D(filters=self.n_h, kernel_size=i+1, kernel_regularizer=l2_reg))
        self._register_trainables(self.convs_h)

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.dense_0 = tf.keras.layers.Dense(self.d, activation=self.act_mlp, input_shape=(self.d * self.n_v + self.n_h,), kernel_regularizer=l2_reg)
        self._register_trainable(self.dense_0)

        self.dense_1_W = tf.keras.layers.Embedding(self.n_items, 2 * self.d, embeddings_regularizer=l2_reg)
        self._register_trainable(self.dense_1_W)

        self.dense_1_b = tf.keras.layers.Embedding(self.n_items, 1)
        self._register_trainable(self.dense_1_b)

        self._sampler = ListSampler(self.interaction_dataset, ['uid'], neg_ratio=neg_ratio, n_targets=self.T,
                                    interaction_threshold=self.interaction_threshold, negative_ids_col='iid',
                                    min_positive_records=self.L, max_positive_records=self.L,
                                    sort_column=self.sort_column, seed=self.seed)

    def _sample_batch(self, batch_size, **kwds):
        uids, iids_before, iids_after = [], [], []
        for (pos_user_records, target_user_records, negative_iids) in self._sampler.sample_group_records(batch_size):
            uids.append(int(pos_user_records[0]['uid']))
            iids_before.append([int(record['iid']) for record in pos_user_records])
            iids_after.append([int(record['iid']) for record in target_user_records] + negative_iids)

        return uids, iids_before, iids_after

    def _predict_batch(self, batch_samples, **kwds):
        predictions, desired_values = [], []

        uids, iids_before, iids_after = batch_samples
        for batch_target_preds in self._predict_batch_aux(uids, iids_before, iids_after, training=True):
            predictions.append(tf.nn.sigmoid(batch_target_preds))
            desired_preds = [1.] * self.T + [0.] * (self.T * self._sampler.neg_ratio)
            desired_values.append(tf.convert_to_tensor(desired_preds))

        return predictions, desired_values

    def _predict_batch_aux(self, uid, iids_before, iids_after, training=False):
        # compute embeddings
        items_embeddings = self.item_embeddings(tf.convert_to_tensor(iids_before))
        user_embeddings = self.user_embeddings(tf.convert_to_tensor(uid))

        # compute convolutions
        out_v = tf.reshape(self.conv_v(items_embeddings), [len(uid), -1])

        out_h = []
        for conv_h in self.convs_h:
            conv_output = self.act_h(conv_h(items_embeddings))
            out_h.append(tf.squeeze(tf.nn.max_pool1d(conv_output, self.n_h, self.n_h, 'SAME'), 1))

        out_h = tf.concat(out_h, 1)
        concat_out = tf.concat([out_v, out_h], 1)

        # compute feedforward
        out_dense = self.dense_0(self.dropout(concat_out, training=training))
        concat_dense_1_input = tf.expand_dims(tf.concat([out_dense, user_embeddings], axis=1), 1)  # (B, 1, 2*d)

        iids_after = tf.convert_to_tensor(iids_after)
        w_ = self.dense_1_W(iids_after)  # (B, targets, 2*d)
        b_ = self.dense_1_b(iids_after)  # (B, targets, 1)
        return tf.reduce_sum(tf.multiply(concat_dense_1_input, w_), 2) + tf.squeeze(b_, 2)

    def _compute_batch_loss(self, predictions, desired_values, **kwds):
        return self._loss(desired_values, predictions)

    def _predict(self, uid, iid, **kwds):
        raise NotImplementedError('This model does not support point-based predictions.')

    def _rank(self, uid, iids, n, novelty):
        user_records = self.interaction_dataset.select(f'uid == {uid}').values_list()

        if self.sort_column in self.interaction_dataset.columns:
            user_records.sort(key=lambda x: x[self.sort_column])

        user_iids = [record['iid'] for record in user_records]

        all_items = list(range(self.n_items))
        predictions = self._predict_batch_aux([uid], [user_iids[-self.L:]], [all_items])[0]
        item_preds = list(zip(predictions, all_items))

        if novelty:
            user_iids_set = set(user_iids)
            novel_user_iids = list(filter(lambda x: x not in user_iids_set, iids))
            item_preds = [(pred, item) for pred, item in item_preds if item in novel_user_iids]
            return nlargest(n, item_preds)

        return nlargest(n, item_preds)
