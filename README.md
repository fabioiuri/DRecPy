[![GitHub version](https://badge.fury.io/py/DRecPy.svg)]()
[![Documentation Status](https://readthedocs.org/projects/drecpy/badge/?version=latest)](https://drecpy.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.com/fabioiuri/DRecPy.svg?branch=master)](https://travis-ci.com/fabioiuri/DRecPy)

# DRecPy

Table of Contents
-----------------

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Implemented Models](#implemented-models)
5. [Benchmarks](#benchmarks)
6. [License](#license)
7. [Contributors](#contributors)
8. [Development Status](#development-status)

Introduction
------------

DRecPy is a Python framework that makes building deep learning based recommender systems easier, 
by making available various tools to develop and test new models.

The main key features DRecPy provides are listed bellow:
- Support for **in-memory and out-of-memory data sets**, by using an intermediary data structure called 
InteractionDataset.
- **Auto Internal to raw id conversion** (identifiers present on the provided data sets): so even if your data set
contains identifiers that are not continuous integers, a mapping will be built automatically: if 
you're using an already built model you won't need to use internal ids; 
otherwise, if you're developing a model, you won't need to use raw ids.
- Well defined **workflow for model building**.
- **Data set splitting techniques** adjusted for the distinct nature of data sets dedicated for 
recommender systems.
- **Sampling techniques** for point based and list based models.
- **Evaluation processes** for predictive models, as well as for learn-to-rank models.
- **Support for multi-column data sets**, i.e. not being limited to (user, item, rating) triples.
- Automatic **plot generation for loss values during model training**, as well as test scores during
model evaluation.
- **All methods with stochastic factors receive a seed parameter**, in order to allow result reproducibility.

For more information about the framework and its components, please visit the [documentation page](https://drecpy.readthedocs.io/).

Installation
------------

With pip:

    $ pip install drecpy

If you can't get the latest version from PyPi:

    $ pip install git+https://github.com/fabioiuri/DRecPy

Or directly by cloning the Git repo:

    $ git clone https://github.com/fabioiuri/DRecPy
    $ cd DRecPy
    $ python setup.py install
    
#### Update Version

If you want to update to the newest DRecPy version, use:

    $ pip install drecpy --upgrade
 

Getting Started
---------------
Here's an example script using one of the implemented recommenders (CDAE), to train, with a validation set,  and evaluate
its ranking performance on the MovieLens 100k data set.
```python
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
ds_train, ds_val = leave_k_out(ds_train, k=1, min_user_interactions=10, seed=0)


def epoch_callback_fn(model):
    return {'val_' + metric: v for metric, v in
            ranking_evaluation(model, ds_val, n_pos_interactions=1, n_neg_interactions=100,
                               generate_negative_pairs=True, k=10, verbose=False, seed=10,
                               metrics={'HR': (hit_ratio, {}), 'NDCG': (ndcg, {})}).items()}


start_train = time.time()
cdae = CDAE(hidden_factors=50, corruption_level=0.2, loss='bce', seed=10)
cdae.fit(ds_train, learning_rate=0.001, reg_rate=0.001, epochs=50, batch_size=64, neg_ratio=5,
         epoch_callback_fn=epoch_callback_fn, epoch_callback_freq=10)
print("Training took", time.time() - start_train)

print(ranking_evaluation(cdae, ds_test, k=[1, 5, 10], novelty=True, n_pos_interactions=1,
                         n_neg_interactions=100, generate_negative_pairs=True, seed=10,
                         max_concurrent_threads=4, verbose=True))
```

**Output**:

```
[CDAE] Max. interaction value: 5
[CDAE] Min. interaction value: 0
[CDAE] Interaction threshold value: 0.001
[CDAE] Number of unique users: 943
[CDAE] Number of unique items: 1680
[CDAE] Number of training points: 89627
[CDAE] Sparsity level: approx. 94.3426%
[CDAE] Creating auxiliary structures...
[CDAE] Model fitted.
Training took 387.92349123954773

{'P@1': 0.1103, 'P@5': 0.0757, 'P@10': 0.0536, 'R@1': 0.1103, 'R@5': 0.3786, 'R@10': 0.5355, 
'HR@1': 0.1103, 'HR@5': 0.3786, 'HR@10': 0.5355, 'NDCG@1': 0.1103, 'NDCG@5': 0.2482, 'NDCG@10': 0.2987, 
'RR@1': 0.1103, 'RR@5': 0.2054, 'RR@10': 0.2261, 'AP@1': 0.1103, 'AP@5': 0.2054, 'AP@10': 0.2261}


```

**Generated Plots**:

- Training

![CDAE Training Performance](https://github.com/fabioiuri/DRecPy/blob/development/examples/images/cdae_validation_training.png?raw=true)

- Evaluation

![CDAE Evaluation Performance](https://github.com/fabioiuri/DRecPy/blob/development/examples/images/cdae_validation_evaluation.png?raw=true)

More quick and easy examples are available [here](https://github.com/fabioiuri/DRecPy/tree/master/examples).

Implemented Models
------------------
| Recommender Type |   Name    |
|:----------------:|:---------:|
| Learn-to-rank    | [CDAE (Collaborative Denoising Auto-Encoder)](https://drecpy.readthedocs.io/en/latest/api_docs/DRecPy.Recommender.html#module-DRecPy.Recommender.cdae) |
| Learn-to-rank    | [DMF (Deep Matrix Factorization)](https://drecpy.readthedocs.io/en/latest/api_docs/DRecPy.Recommender.html#module-DRecPy.Recommender.dmf)              |

#### Implemented Baselines (non deep learning based) 
| Recommender Type |   Name    |
|:----------------:|:---------:|
| Predictive       | [User/Item KNN](https://drecpy.readthedocs.io/en/latest/api_docs/DRecPy.Recommender.Baseline.html#drecpy-recommender-baseline-knn-module) |

Benchmarks
----------

TODO

License
-------

Check [LICENCE.md](https://github.com/fabioiuri/DRecPy/blob/master/LICENSE.md).

Contributors
------------

This work was conducted under the supervision of Prof. Francisco M. Couto, and during the initial development phase the project was financially supported by a FCT research scholarship UID/CEC/00408/2019, under the research institution LASIGE, from the Faculty of Sciences, University of Lisbon.

Development Status
------------------

Project in pre-alpha stage.

Planned work:
- Wrap up missing documentation
- Implement more models
- Implement list-wise sampling strategy
- Refine and clean unit tests

If you have any bugs to report or update suggestions, you can use DRecPy's [github issues page](https://github.com/fabioiuri/DRecPy/issues) or email me directly to fabioiuri@live.com.