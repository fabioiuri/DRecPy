from DRecPy.Recommender import RecommenderABC
import tensorflow as tf
from DRecPy.Sampler import PointSampler
from heapq import nlargest
import numpy as np


class Test(RecommenderABC):

    def __init__(self, n_supported_items=5, A=256, B=128, C=64, **kwds):
        super(Test, self).__init__(**kwds)

        self.A = A
        self.B = B
        self.C = C
        self.n_supported_items = n_supported_items

    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        self._user_emb = tf.keras.Sequential(tf.keras.layers.Embedding(self.n_users, self.A))
        self._item_emb = tf.keras.Sequential(tf.keras.layers.Embedding(self.n_items, self.A))

        self._hidden_models = []
        for i in range(self.n_supported_items):
            self._hidden_models.append(
                tf.keras.Sequential(tf.keras.layers.Dense(self.B, activation=tf.nn.relu, input_shape=(2 * self.A,)))
            )

        self._out_model = tf.keras.Sequential()
        self._out_model.add(tf.keras.layers.Dense(self.C, activation=tf.nn.relu, input_shape=(self.n_supported_items * self.B,)))
        self._out_model.add(tf.keras.layers.Dense(self.n_supported_items, activation=tf.nn.softmax))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        interaction_threshold = (self.max_interaction + self.min_interaction) / 2
        self._log(f'Interaction threshold for positive pairs: {interaction_threshold}')

        self._positive_sampler = PointSampler(self.interaction_dataset, 0, interaction_threshold, self.seed)

    def _do_batch(self, **kwds):
        uid, _ = self._positive_sampler.sample_positive()
        user_records = self._rng.sample(self.interaction_dataset.select(f'uid == {uid}').values_list(),
                                        self.n_supported_items)

        best_ranking = sorted(user_records, key=lambda rec: -rec['interaction'])
        for i, user_record in enumerate(user_records):
            for j, best in enumerate(best_ranking):
                if best == user_record:
                    user_record['best_idx'] = j
                    break

        # print('best_ranking', [(u['interaction'], u['best_idx']) for u in user_records])

        best_ranking = [user_record['best_idx'] for user_record in user_records]

        # print('best_ranking',best_ranking)

        with tf.GradientTape() as tape:
            pred = self._model_predict(uid, [r['iid'] for r in user_records])
            loss = self._compute_loss(pred, best_ranking)
            # print('loss', loss)

            out_grads = tape.gradient(loss, [self._out_model.trainable_variables,
                                             *[model.trainable_variables for model in self._hidden_models],
                                             self._item_emb.trainable_variables,
                                             self._user_emb.trainable_variables])
            # print('out_grads', out_grads)
            # print()

        self.optimizer.apply_gradients(zip(out_grads[0], self._out_model.trainable_variables))
        for i, hidden_model in enumerate(self._hidden_models):
            self.optimizer.apply_gradients(zip(out_grads[i + 1], hidden_model.trainable_variables))
        self.optimizer.apply_gradients(zip(out_grads[-2], self._item_emb.trainable_variables))
        self.optimizer.apply_gradients(zip(out_grads[-1], self._user_emb.trainable_variables))

        return loss

    def _model_predict(self, uid, iids):
        user_rep = self._user_emb(np.int(uid))
        joint_reps = []
        for iid, hidden_model in zip(iids, self._hidden_models):
            item_rep = self._item_emb(np.int(iid))
            joint_rep = hidden_model(tf.convert_to_tensor([tf.concat([user_rep, item_rep], axis=0)]))[0]
            joint_reps.append(joint_rep)

        stacked_rep = tf.concat(joint_reps, axis=0)
        return self._out_model(tf.convert_to_tensor([stacked_rep]))[0]

    def _compute_loss(self, pred_ranking, best_ranking):
        #print('pred_ranking_normalized', tf.nn.l2_normalize(tf.convert_to_tensor([pred_ranking], dtype=tf.float32)))
        #print('best_ranking_normalized', tf.nn.l2_normalize(tf.convert_to_tensor([best_ranking], dtype=tf.float32)))
        return 1 - tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(tf.convert_to_tensor([pred_ranking], dtype=tf.float32)),
                                             tf.nn.l2_normalize(tf.convert_to_tensor([best_ranking], dtype=tf.float32))))

    def _predict(self, uid, iid, **kwds): # todo
        return 0

    def _rank(self, uid, iids, n, novelty): # todo
        if novelty:
            rated_items = self.interaction_dataset.select(f'uid == {uid}').values_list(columns=['iid'], to_list=True) # todo: check if uid iid pair exist instead of this
            iids = list(set(iids).difference(set(rated_items)))

        if len(iids) < self.n_supported_items:
            # TODO: fill missing with zeroes? must train this case as well probably
            raise Exception(f'not supported yet - len(iids) >= n_supported_items ({len(iids)} >= {self.n_supported_items})')

        curr_iid_bucket = iids[:self.n_supported_items]
        iids = iids[self.n_supported_items:]
        while len(iids) > 0:
            pred_ranking = self._model_predict(uid, curr_iid_bucket)
            bucket_item_rankings = [(iid, pred_position) for iid, pred_position in zip(curr_iid_bucket, pred_ranking)]
            worst_bucket_item = sorted(bucket_item_rankings, key=lambda x: x[1])[-1][0]
            curr_iid_bucket.remove(worst_bucket_item)
            curr_iid_bucket.append(iids.pop())

        pred_ranking = self._model_predict(uid, curr_iid_bucket)
        pred_list = [(-pred_position, iid) for pred_position, iid in zip(pred_ranking, curr_iid_bucket)]

        return nlargest(n, pred_list)
