import tensorflow as tf

from models.item_ranking.cdae import CDAE
from utils.evaluation.RankingMetrics import evaluate


class ModifiedCDAE(CDAE):
    def __init__(self, sess, num_user, num_item, nn_factors=None, **kwds):
        super(ModifiedCDAE, self).__init__(sess, num_user, num_item, **kwds)
        self.nn_factors = nn_factors if nn_factors is not None else [512, 1024]

    def build_network(self, hidden_neuron=500, corruption_level=0):
        super(ModifiedCDAE, self).build_network(corruption_level=corruption_level)
        _W = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.num_item, hidden_neuron], stddev=0.01))
        _W_prime = tf.compat.v1.Variable(tf.compat.v1.random_normal([hidden_neuron, self.num_item], stddev=0.01))
        _V = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.num_user, hidden_neuron], stddev=0.01))

        b = tf.compat.v1.Variable(tf.compat.v1.random_normal([hidden_neuron], stddev=0.01))
        b_prime = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.num_item], stddev=0.01))

        self.nn_factors.append(self.num_item)
        nn_weights = [tf.compat.v1.Variable(tf.compat.v1.random_normal([self.num_item, self.nn_factors[0]], stddev=0.01))]
        for i in range(1, len(self.nn_factors)):
            nn_weights.append(tf.compat.v1.Variable(tf.compat.v1.random_normal([self.nn_factors[i-1], self.nn_factors[i]], stddev=0.01)))
        nn_biases = [tf.compat.v1.Variable(tf.compat.v1.random_normal([factor], stddev=0.01)) for factor in self.nn_factors]
        self.final_layer = tf.compat.v1.sigmoid(tf.compat.v1.matmul(self.layer_2, nn_weights[0]) + nn_biases[0])
        for i in range(1, len(self.nn_factors)):
            self.final_layer = tf.compat.v1.sigmoid(tf.compat.v1.matmul(self.final_layer, nn_weights[i]) + nn_biases[i])

        self.loss = - tf.compat.v1.reduce_sum(
            self.rating_matrix * tf.compat.v1.log(self.final_layer) + (1 - self.rating_matrix) * tf.compat.v1.log(1 - self.final_layer)) + \
            self.reg_rate * (tf.compat.v1.nn.l2_loss(_W) + tf.compat.v1.nn.l2_loss(_W_prime) + tf.compat.v1.nn.l2_loss(_V) +
                             tf.compat.v1.nn.l2_loss(b) + tf.compat.v1.nn.l2_loss(b_prime) +
                             sum([tf.compat.v1.nn.l2_loss(weight) for weight in nn_weights]) +
                             sum([tf.compat.v1.nn.l2_loss(bias) for bias in nn_biases]))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def test(self):
        self.reconstruction = self.sess.run(self.final_layer, feed_dict={self.corrupted_rating_matrix: self.train_data,
                                                                         self.user_id: range(self.num_user)})
        evaluate(self)
