.. DRecPy documentation master file, created by
   sphinx-quickstart on Mon Sep 16 00:28:55 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DRecPy's documentation!
===================================

.. toctree::
   :caption: User Guide
   :hidden:

   user_guide/installation
   user_guide/getting_started
   user_guide/creating_recommender


.. toctree::
   :maxdepth: 2
   :caption: API Documentation
   :hidden:

   api_docs/DRecPy.Dataset
   api_docs/DRecPy.Recommender
   api_docs/DRecPy.Sampler
   api_docs/DRecPy.Evaluation

|GitHub Version| |Documentation Status| |License: MIT| |Travis CI|

Table of Contents
-----------------

1. `Introduction <#introduction>`__
2. `Installation <#installation>`__
3. `Getting Started <#getting-started>`__
4. `Implemented Models <#implemented-models>`__
5. `Benchmarks <#benchmarks>`__
6. `License <#license>`__
7. `Contributors <#contributors>`__
8. `Development Status <#development-status>`__

Introduction
------------

DRecPy is a Python framework that makes building deep learning based
recommender systems easier, by making available various tools to develop
and test new models.

The main key features DRecPy provides are listed bellow:

- Support for **in-memory and out-of-memory data sets**, by using an intermediary data structure called InteractionDataset.

- **Auto Internal to raw id conversion** (identifiers present on the provided data sets): so even if your data set contains identifiers that are not continuous integers, a mapping will be built automatically: if you're using an already built model you won't need to use internal ids; otherwise, if you're developing a model, you won't need to use raw ids.

- Well defined **workflow for model building**.

- **Data set splitting techniques** adjusted for the distinct nature of data sets dedicated for recommender systems.

- **Sampling techniques** for point based and list based models.

- **Evaluation processes** for predictive models, as well as for learn-to-rank models.

- **Support for multi-column data sets**, i.e. not being limited to (user, item, rating) triples.

- Automatic **plot generation for loss values during model training**, as well as test scores during model evaluation.

- **All methods with stochastic factors receive a seed parameter**, in order to allow result reproducibility.

For more information about the framework and its components, please
visit the `documentation page <https://drecpy.readthedocs.io/>`__.

Installation
------------

With pip:

::

    $ pip install drecpy

If you can't get the latest version from PyPi:

::

    $ pip install git+https://github.com/fabioiuri/DRecPy

Or directly by cloning the Git repo:

::

    $ git clone https://github.com/fabioiuri/DRecPy
    $ cd DRecPy
    $ python setup.py install

Getting Started
---------------

Here's an example script using one of the implemented recommenders
(CDAE), to train, with a validation set, and evaluate its ranking
performance on the MovieLens 100k data set.

.. code:: python

    from DRecPy.Recommender import CDAE
    from DRecPy.Dataset import get_train_dataset
    from DRecPy.Dataset import get_test_dataset
    from DRecPy.Evaluation.Processes import ranking_evaluation
    from DRecPy.Evaluation.Splits import leave_k_out
    from DRecPy.Evaluation.Metrics import ndcg
    from DRecPy.Evaluation.Metrics import hit_ratio
    import time


    ds_train = get_train_dataset('ml-100k')
    ds_test = get_test_dataset('ml-100k')
    ds_train, ds_val = leave_k_out(ds_train, k=1, min_user_interactions=10)


    def epoch_callback_fn(model):
        return {'val_' + metric: v for metric, v in
                ranking_evaluation(model, ds_val, n_pos_interactions=1, n_neg_interactions=100,
                                   generate_negative_pairs=True, k=10, verbose=False, seed=10,
                                   metrics={'HR': (ndcg, {}), 'NDCG': (hit_ratio, {})}).items()}


    start_train = time.time()
    cdae = CDAE(hidden_factors=50, corruption_level=0.2, loss='bce', seed=10)
    cdae.fit(ds_train, learning_rate=0.001, reg_rate=0.001, epochs=80, batch_size=64, neg_ratio=5,
             epoch_callback_fn=epoch_callback_fn, epoch_callback_freq=20)
    print("Training took", time.time() - start_train)

    print(ranking_evaluation(cdae, ds_test, k=[1, 5, 10], novelty=True, n_pos_interactions=1,
                             n_neg_interactions=100, generate_negative_pairs=True, seed=10,
                             max_concurrent_threads=4, verbose=True))

**Output**:

::

    [CDAE] Max. interaction value: 5
    [CDAE] Min. interaction value: 0
    [CDAE] Interaction threshold value: 0
    [CDAE] Number of unique users: 943
    [CDAE] Number of unique items: 1680
    [CDAE] Number of training points: 89627
    [CDAE] Sparsity level: approx. 94.3426%
    [CDAE] Creating auxiliary structures...
    [CDAE] Model fitted.
    Training took 1620.2718272209167

    {'P@1': 0.141, 'P@5': 0.0793, 'P@10': 0.0591, 'R@1': 0.141, 'R@5': 0.3966, 'R@10': 0.5907,
    'HR@1': 0.141, 'HR@5': 0.3966, 'HR@10': 0.5907, 'NDCG@1': 0.141, 'NDCG@5': 0.2701, 'NDCG@10': 0.3327,
    'RR@1': 0.141, 'RR@5': 0.2286, 'RR@10': 0.2543, 'AP@1': 0.141, 'AP@5': 0.2286, 'AP@10': 0.2543}

**Generated Plots**:

-  Training

.. figure:: https://github.com/fabioiuri/DRecPy/blob/development/examples/images/cdae_validation_training.png?raw=true
   :alt: CDAE Training Performance

   CDAE Training Performance

-  Evaluation

.. figure:: https://github.com/fabioiuri/DRecPy/blob/development/examples/images/cdae_validation_evaluation.png?raw=true
   :alt: CDAE Evaluation Performance

   CDAE Evaluation Performance

More quick and easy examples are available `here <https://github.com/fabioiuri/DRecPy/tree/master/examples>`__.


Implemented Models
------------------

+--------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Recommender Type   | Name                                                                                                                                                        |
+====================+=============================================================================================================================================================+
| Learn-to-rank      | `CDAE (Collaborative Denoising Auto-Encoder) <https://drecpy.readthedocs.io/en/latest/api_docs/DRecPy.Recommender.html#module-DRecPy.Recommender.cdae>`__   |
+--------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Learn-to-rank      | `DMF (Deep Matrix Factorization) <https://drecpy.readthedocs.io/en/latest/api_docs/DRecPy.Recommender.html#module-DRecPy.Recommender.dmf>`__                |
+--------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+

Implemented Baselines (non deep learning based)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+--------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Recommender Type   | Name                                                                                                                                           |
+====================+================================================================================================================================================+
| Predictive         | `User/Item KNN <https://drecpy.readthedocs.io/en/latest/api_docs/DRecPy.Recommender.Baseline.html#drecpy-recommender-baseline-knn-module>`__   |
+--------------------+------------------------------------------------------------------------------------------------------------------------------------------------+

Benchmarks
----------

TODO

License
-------

Check
`LICENCE.md <https://github.com/fabioiuri/DRecPy/blob/master/LICENSE.md>`__.

Contributors
------------

This work was conducted under the supervision of Prof. Francisco M.
Couto, and during the initial development phase the project was
financially supported by a FCT research scholarship UID/CEC/00408/2019,
under the research institution LASIGE, from the Faculty of Sciences,
University of Lisbon.

Development Status
------------------

Project in pre-alpha stage.

Planned work:

- Wrap up missing documentation

- Implement more models

- Implement list-wise sampling strategy

- Refine and clean unit tests

If you have any bugs to report or update suggestions, you can use
DRecPy's `github issues
page <https://github.com/fabioiuri/DRecPy/issues>`__ or email me
directly to fabioiuri@live.com.

.. |GitHub Version| image:: https://badge.fury.io/py/DRecPy.svg
.. |Documentation Status| image:: https://readthedocs.org/projects/drecpy/badge/?version=latest
   :target: https://drecpy.readthedocs.io/en/latest/?badge=latest
.. |License: MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
.. |Travis CI| image:: https://travis-ci.com/fabioiuri/DRecPy.svg?branch=master
    :target: https://travis-ci.com/fabioiuri/DRecPy