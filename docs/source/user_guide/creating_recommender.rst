***************************
Creating Novel Recommenders
***************************

From scratch
============
DRecPy allows you to easily implement novel recommender models without the need to worry about data handling, id conversion, workflow management and weight updates.

Deep Learning-based Recommenders
--------------------------------
To create a DL-based recommender you should create a class that extends the `RecommenderABC` class, and implements at least the required abstract methods: `_pre_fit()`, `_sample_batch()`, `_predict_batch()`, `_compute_batch_loss()`, `_compute_reg_loss()`  and `_predict()`.

The `_pre_fit()` method should:

- Create model structure (e.g. neural network layer structure);
- Initialize model weights;
- Register the model weights as trainable variables (using the `_register_trainable(var)` or `_register_trainables(vars)` method);

The `_sample_batch()` method should sample `N` data points from the training data and return them. This step can also be used if your model makes some custom data preprocessing during the training phase. The return value of this function is passed to the `_predict_batch()` method.

The `_predict_batch()` method should compute predictions for each of the sampled training data points. The return of this method should be a tuple containing a list of predictions and a list of desired values (in this exact order).

The `_compute_batch_loss()` method should compute and return the loss value associated with the given predictions and desired values. This loss must be differentiable with respect to all training variables, so that weight updates can be made.

The `_compute_reg_loss()` method should compute and return the regularization loss value associated with the trainable variables.

Finally, the `_predict()` method should compute a single prediction, for the provided user id and item id.

If the `_rank()` method is not implemented, and if you call `rank()` on the new model, the `_predict()` method will be used to score each item.
Similarly, if the `_recommend()` method is not implemented, and if you call `recommend()`, the `rank()` will also be used to rank all existent items and return the top `N`.

Usually, the `_rank()` and `_recommend()` methods are only implemented when there's an alternative and more efficient way to compute these values, not relying on the `_predict()` method.

Here's a basic example of a deep learning-based recommender, with only one trainable variable:

.. literalinclude:: ../../../examples/custom_deep_recommender.py
    :caption: From file ``examples/custom_deep_recommender.py``
    :name: custom_deep_recommender.py


Some other **important notes**:

- Do not forget to call the `__init__()` method of the super class, on the new model `__init__()`;
- The `_pre_fit()` method should register all trainable variables via calls to the `_register_trainable(var)` or `_register_trainables(vars)` methods;
- The `_compute_batch_loss()` return value must be differentiable with respect to all trainable variables;
- If your model depends on custom random processes, such as weight initialization or id sampling, always use random generators that are created using the self.seed attribute. A seeded random generator object is already provided via the **self._rng** attribute. If a seed argument is provided, both self._rng and all tensorflow.random methods are seeded with the given value.


Non-Deep Learning-based Recommenders
------------------------------------
To create a non-DL-based recommender you should create a class that extends the `RecommenderABC` class, and implements at least the required abstract methods: `_pre_fit()`, `_sample_batch()`, `_predict_batch()`, `_compute_batch_loss()`,  `_compute_reg_loss()` and `_predict()`.

Note that in this case, the implementation of the `_sample_batch()`, `_predict_batch()`, `_compute_batch_loss()` and  `_compute_reg_loss()` is irrelevant, since they will never be called.

The `_pre_fit()` method should create the required data structures and do the computations to completely fit the model, because no batch training will be applied. In this case, **no trainable variable can be registered**.

All other methods such as the `_predict()`, `_rank()` and `_recommend()`, follow the same guidelines.

An example of an implemented non-deep learning-based recommender, follows bellow:

.. literalinclude:: ../../../examples/custom_non_deep_recommender.py
    :caption: From file ``examples/custom_non_deep_recommender.py``
    :name: custom_non_deep_recommender.py

Extending existing models
=========================
To extend an existent recommender, one should:

- Create a subclass of the original recommender;
- Override the `__init__()` method, by first calling the original recommender `__init__()`, followed by instructions with specific logic for the extended recommender;
- If new weights are introduced by this extension, override the `_pre_fit()` method and call its original, followed by the initialization of the new weights and registering them as trainable variables (via `_register_trainable(var)` or `_register_trainables(vars)` method calls).
- If there are changes to the batch sampling workflow, override the `_sample_batch()` method and call its original (if some of its logic can be reused), and then apply the custom logic;
- When there are changes on the way predictions are made, override the `_predict_batch()` and call its original (if some of its logic can be reused), followed by adding the custom prediction logic to the existent predictions. Sometimes, if the original model has independent logic on the `_predict()` that does not use the `_predict_batch()`, you might need to override it too.
- If there are changes to the way the predictive loss function is computed, override the `_compute_batch_loss()` and adapt it to return the new predictive loss value from the provided predictions and expected values.
- If there are changes to the way the regularization loss function is computed, maybe due to having additional trainable variables, override the `_compute_reg_loss()` and adapt it to return the new regularization loss value from the current state of the network's weights.

A very simple example of extending an existing model is shown bellow, which is a modification of the DMF (Deep Matrix Factorization) recommender:

.. literalinclude:: ../../../examples/extending_recommender_dmf.py
    :caption: From file ``examples/extending_recommender_dmf.py``
    :name: extending_recommender_dmf.py