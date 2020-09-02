from DRecPy.Recommender import CDAE
import tensorflow as tf


class ModifiedCDAE(CDAE):
    def __init__(self, nn_factors=None, **kwds):
        super(ModifiedCDAE, self).__init__(**kwds)
        self.nn_factors = nn_factors if nn_factors is not None else [512, 1024]

    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        super(ModifiedCDAE, self)._pre_fit(learning_rate, neg_ratio, reg_rate, **kwds)
        self.nn_factors.append(self.n_items)

        l2_reg = tf.keras.regularizers.l2(reg_rate)
        self.nn = tf.keras.Sequential()
        self.nn.add(tf.keras.layers.Dense(self.nn_factors[0], activation=tf.nn.relu, input_shape=(self.n_items,),
                                          kernel_regularizer=l2_reg, autocast=False))
        for factor in self.nn_factors[1:]:
            self.nn.add(tf.keras.layers.Dense(factor, activation=tf.nn.relu, kernel_regularizer=l2_reg))
        self._register_trainable(self.nn)

    def _reconstruct_for_training(self, uid):
        prediction_vector, desired_vector = super(ModifiedCDAE, self)._reconstruct_for_training(uid)
        return self.nn(prediction_vector), desired_vector

    def _compute_reg_loss(self, reg_rate, batch_size, trainable_models, trainable_layers, trainable_weights, **kwds):
        return super(ModifiedCDAE, self)._compute_reg_loss(reg_rate, batch_size, [], trainable_layers, trainable_weights, **kwds) + tf.math.add_n(trainable_models[0].losses) / batch_size
