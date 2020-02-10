***************************
Creating Novel Recommenders
***************************

From scratch
============
DRecPy allows you to easily implement novel recommender models without the need to worry about data handling, id conversion and workflow management.

To do so, you should create a class that extends the `RecommenderABC` class, and implements at least the required abstract methods: `_pre_fit()`, `_do_batch()` and `_predict()`.
Other methods, such as the `_rank()` and the `_recommend()`, can be implemented in order to have the desired behaviour.

If the `_rank()` method is not implemented, and if you call `rank()` on the new model, the `_predict()` method will be used to score each product.
Similarly, if the `recommend()` method is not implemented, and if you call the `recommend()`, the `predict()` will also be called for each product in order to sort all products accordingly to that score.

Here's the most basic example for creating a new model:

.. literalinclude:: ../../../examples/custom_recommender.py
    :caption: From file ``examples/custom_recommender.py``
    :name: custom_recommender.py


Some other **important notes**:

- Do not forget to call the `__init__()` method of the super class, on the new model `__init__()`;
- The `_do_batch()` method should return the loss value computed during the respective batch evaluation. If this is not done and `verbose = True`, then an exception will be raised.
- If your model depends on random processes, such as weight initialization or id sampling, always use random generators that are created using the self.seed attribute. A random generator object is already provided via the **self._rng** attribute. If a seed argument is provided, both self._rng and all tensorflow.random methods are seeded with the given value.

A real example that uses a custom `_rank()` method is shown bellow:

.. literalinclude:: ../../../DRecPy/Recommender/cdae.py
    :caption: From file ``DRecPy/Recommender/cdae.py``
    :name: Recommender/cdae.py

Extending existing models
=========================
...
