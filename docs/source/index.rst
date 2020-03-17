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
visit the `documentation page`_.

Installation
------------

With pip:

::

   $ pip install drecpy

If you can’t get the latest version from PyPi:

::

   $ pip install git+https://github.com/fabioiuri/DRecPy

Or directly by cloning the Git repo:

::

   $ git clone https://github.com/fabioiuri/DRecPy
   $ cd DRecPy
   $ python setup.py install

Getting Started
---------------

Here’s an example script using one of the implemented recommenders
(CDAE), to train and evaluate its ranking performance on the MovieLens
100k data set.

.. code:: python

   from DRecPy.Recommender import CDAE
   from DRecPy.Dataset import get_train_dataset
   from DRecPy.Dataset import get_test_dataset
   from DRecPy.Evaluation import ranking_evaluation
   from DRecPy.Evaluation import predictive_evaluation
   import time

   ds_train = get_train_dataset('ml-100k', verbose=False)
   ds_test = get_test_dataset('ml-100k', verbose=False)

   start_train = time.time()
   cdae = CDAE(min_interaction=0, seed=10)
   cdae.fit(ds_train, epochs=50)
   print("Training took", time.time() - start_train)

   print(ranking_evaluation(cdae, ds_test, n_test_users=100, seed=10))
   print(predictive_evaluation(cdae, ds_test, skip_errors=True))

**Output**:

::

   [CDAE] Max. interaction value: 5
   [CDAE] Min. interaction value: 1
   [CDAE] Number of unique users: 943
   [CDAE] Number of unique items: 1680
   [CDAE] Number of training points: 90570
   [CDAE] Sparsity level: approx. 94.2831%
   [CDAE] Creating auxiliary structures...
   [CDAE] Model fitted.
   Training took 25.366847276687622

   {'P@10': 0.061, 'R@10': 0.61, 'HR@10': 0.61, 'NDCG@10': 0.3517, 'RR@10': 0.2734, 'AP@10': 0.2734}
   {'RMSE': 3.1898, 'MSE': 10.1745}

More quick and easy examples are available `here`_.

Implemented Models
------------------

================ ==============================================
Recommender Type Name
================ ==============================================
Learn-to-rank    `CDAE (Collaborative Denoising Auto-Encoder)`_
Learn-to-rank    `DMF (Deep Matrix Factorization)`_
================ ==============================================

Implemented Baselines (non deep learning based)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

================ ================
Recommender Type Name
================ ================
Predictive       `User/Item KNN`_
================ ================

Benchmarks
----------

TODO

License
-------

Check `LICENCE.md`_.

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
DRecPy's `github issues page`_ or email me directly to
fabioiuri@live.com.

.. _documentation page: https://drecpy.readthedocs.io/
.. |GitHub version| image:: https://badge.fury.io/py/DRecPy.svg
   :target:
.. |Documentation Status| image:: https://readthedocs.org/projects/drecpy/badge/?version=latest
   :target: https://drecpy.readthedocs.io/en/latest/?badge=latest
.. |License: MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
.. _here: https://github.com/fabioiuri/DRecPy/tree/master/examples
.. _CDAE (Collaborative Denoising Auto-Encoder): https://drecpy.readthedocs.io/en/latest/api_docs/DRecPy.Recommender.html#module-DRecPy.Recommender.cdae
.. _DMF (Deep Matrix Factorization): https://drecpy.readthedocs.io/en/latest/api_docs/DRecPy.Recommender.html#module-DRecPy.Recommender.dmf
.. _User/Item KNN: https://drecpy.readthedocs.io/en/latest/api_docs/DRecPy.Recommender.Baseline.html#drecpy-recommender-baseline-knn-module
.. _LICENCE.md: https://github.com/fabioiuri/DRecPy/blob/master/LICENSE.md
.. _github issues page: https://github.com/fabioiuri/DRecPy/issues