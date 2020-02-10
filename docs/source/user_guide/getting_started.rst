***************
Getting Started
***************

Basic usage
===============

Training a model
----------------
Recommenders built using the DRecPy framework follow the usual method definitions: `fit()` to fit the model to the provided data, and `predict()`, `rank()` or `recommend()` to provide predictions.
Once trained, in order to evaluate a model, one can build custom evaluation processes, or can use the builtin ones, which are defined on the :ref:`evaluation_docs`.

Here's a quick example of training the CDAE recommender with the MovieLens-100k data set on 100 epochs, and evaluating the ranking performance on 100 test users. Node that a seed parameter is passed through when instantiating the CDAE object, as well as when calling the evaluation process, so that we can have a deterministic pipeline.

.. literalinclude:: ../../../examples/cdae.py
    :caption: From file ``examples/cdae.py``
    :name: cdae.py


Data Set usage
==============

To learn more about the public methods offered by the `InteractionDataset` module, please read the respective api documentation. This section is simply a brief introduction on how to import and make use of data sets.

Importing a built-in data set
-----------------------------
At the moment, DRecPy provides various builtin data sets, such as: the MovieLens (100k, 1M, 10M and 20M) and the Book Crossing data set.
Whenever you're using a builtin data set for the first time, a new folder will be created at your home path called ".DRecPy_data". If you want to provide a custom path for saving these data sets, you can do so by providing the `DATA_FOLDER` environment variable mapping to the intended path.

The example bellow shows how to use a builtin data set and how to manipulate it using the provided methods:

.. literalinclude:: ../../../examples/integrated_datasets.py
    :caption: From file ``examples/integrated_datasets.py``
    :name: integrated_datasets.py


Importing a custom data set
---------------------------
Custom data sets are also supported, and you should provide the path to the csv file as well as the column names and the delimiter.

.. literalinclude:: ../../../examples/custom_datasets.py
    :caption: From file ``examples/custom_datasets.py``
    :name: custom_datasets.py

Note that there are **3 required columns**: user, item and interaction.