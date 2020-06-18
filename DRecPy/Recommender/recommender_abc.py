from abc import ABC
from abc import abstractmethod
from joblib import load
from joblib import dump
from tqdm import tqdm
from DRecPy.Evaluation import LossTracker
from heapq import nlargest
import random
import tensorflow as tf

tf.config.set_soft_device_placement(True)  # automatically choose an existing and supported device to run (GPU, CPU)


class RecommenderABC(ABC):
    """Base recommender abstract class.

    This class implements the skeleton methods required for building a recommender.
    It provides id-abstraction (handles conversion between raw to internal ids - by ),
    auto-exist-check (if a given user/item is known or not) and all private methods
    are called with internal ids only. All public methods are called with raw ids only.

    The following methods are still required to be implemented: _pre_fit(), _do_batch() and _predict().
    Optionally, these methods can be overridden: _rank() and _recommend().

    Args:
        verbose: Optional boolean indicating if the recommender should print progress logs or not.
            Default: False.
        interaction_threshold: An optional integer that is used as the boundary interaction value between positive and
            negative interaction pairs. All values above or equal interaction_threshold are considered positive, and
            all values bellow are considered negative. Default: 0.
        seed (max_rating): Optional integer representing the seed value for the model pseudo-random number generator.
            Default: None.
    """

    def __init__(self, **kwds):
        self.verbose = kwds.get('verbose', True)
        self.min_interaction = None
        self.max_interaction = None
        self.seed = kwds.get('seed', None)

        self.fitted = False
        self.n_users = 0
        self.n_items = 0
        self.n_rows = 0
        self.interaction_threshold = kwds.get('interaction_threshold', 1e-3)
        self.interaction_dataset = None

        self._loss_tracker = None
        self._rng = random.Random(self.seed)
        tf.random.set_seed(self.seed)

    def fit(self, interaction_dataset, epochs=50, batch_size=32, learning_rate=0.001, neg_ratio=5, reg_rate=0.001,
            copy_dataset=False, **kwds):
        """Processes the provided dataframe and builds id-abstraction, infers min. and max. interactions
        (if not passed through constructor) and calls _fit() to fit the current model.

        Args:
            interaction_dataset: A interactionsDataset instance containing the training data.
            epochs: Optional number of epochs to train the model. Default: 50.
            batch_size: Optional number of data points to use for each epoch to train the model. Default: 32.
            learning_rate: Optional decimal representing the learning rate of the model. Default: 0.001.
            neg_ratio: Optional integer that represents the number of negative instances for each positive one.
                Default: 5.
            reg_rate: Optional decimal representing the model regularization rate. Default: 0.01.
            epoch_callback_fn: Optional function that is called, for each epoch_callback_freq, with the model at its
                current state. It receives one argument - the model at its current state - and should return a dict
                mapping each metric's name to the corresponding value. The results will be displayed in a graph at the
                end of the model fit and during the fit process on the logged progress bar description only if verbose
                is set to True.
            epoch_callback_freq: Optional integer representing the frequency in which the epoch_callback_fn is called.
                If epoch_callback_fn is not defined, this parameter is ignored. Default: 5 (called every 5 epochs).
            copy_dataset: Optional boolean indicating weather a copy of the given dataset should be made.
                If set to False, the given dataset instance is used. Default: False.

        Returns:
            None.
        """
        self.interaction_dataset = interaction_dataset
        if copy_dataset:
            self._log('Cloning new dataset instance...')
            self.interaction_dataset = interaction_dataset.__copy__()
        self.interaction_dataset.assign_internal_ids()

        self.min_interaction = self.interaction_dataset.min('interaction')
        if self.min_interaction == 1: self.min_interaction = 0
        self.max_interaction = self.interaction_dataset.max('interaction')

        self.n_users = self.interaction_dataset.count_unique('uid')
        self.n_items = self.interaction_dataset.count_unique('iid')
        self.n_rows = len(self.interaction_dataset)

        self._loss_tracker = LossTracker()

        # Log extra info
        self._log_initial_info()

        self._log('Creating auxiliary structures...')
        self._pre_fit(learning_rate, neg_ratio, reg_rate, **kwds)
        self.fitted = True  # should be able to make predictions after pre fit

        epoch_callback_fn = kwds.get('epoch_callback_fn', None)
        epoch_callback_res, epoch_callback_res_registered = None, True
        epoch_callback_freq = kwds.get('epoch_callback_freq', 5)
        curr_epoch_callback_count = 0

        try:
            _iter = range(1, epochs+1)
            if self.verbose:
                _iter = tqdm(range(1, epochs+1), total=epochs, desc='Fitting model...', position=0, leave=True)
            for e in _iter:
                self._loss_tracker.reset_batch_losses()
                for b in range(1, batch_size+1):
                    loss = self._do_batch(**kwds)
                    if self.verbose:
                        if loss is None: raise Exception('Model\'s ._do_batch() method must return a valid loss obtained during that batch.')
                        self._loss_tracker.add_batch_loss(loss)

                curr_epoch_callback_count -= 1

                if epoch_callback_fn is not None and curr_epoch_callback_count <= 0:
                    curr_epoch_callback_count = epoch_callback_freq
                    epoch_callback_res_registered = False
                    epoch_callback_res = epoch_callback_fn(self)
                    assert type(epoch_callback_res) is dict, \
                        f'The return type of the epoch_callback_fn should be dict, but found {type(epoch_callback_res)}'

                if self.verbose:
                    progress_desc = f'Fitting model... Epoch {e} Avg. Loss: {self._loss_tracker.get_batch_avg_loss():.4f}'
                    if epoch_callback_res is not None:
                        for metric in epoch_callback_res:
                            progress_desc += f' | {metric}: {epoch_callback_res[metric]}'
                            if not epoch_callback_res_registered:
                                self._loss_tracker.add_epoch_callback_result(metric, epoch_callback_res[metric], e)
                        epoch_callback_res_registered = True

                    _iter.set_description(progress_desc)
                    self._loss_tracker.update_epoch_loss()  # update avg training loss

            if self.verbose: self._loss_tracker.display_graph(model_name=self.__class__.__name__)
        except NotImplementedError:
            pass

        self._log('Model fitted.')

    def _log_initial_info(self):
        self._log(f'Max. interaction value: {self.max_interaction}')
        self._log(f'Min. interaction value: {self.min_interaction}')
        self._log(f'Interaction threshold value: {self.interaction_threshold}')
        self._log(f'Number of unique users: {self.n_users}')
        self._log(f'Number of unique items: {self.n_items}')
        self._log(f'Number of training points: {self.n_rows}')
        matrix_size = self.n_users * self.n_items
        sparsity = round(100 * (1 - (self.n_rows / matrix_size)), 4)
        self._log(f'Sparsity level: approx. {sparsity}%')

    @abstractmethod
    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        """Abstract method that should setup all the required structures for fitting the model.
        Should use the provided interaction_dataset to do the training procedure."""
        pass

    @abstractmethod
    def _do_batch(self, **kwds):
        """Abstract method that should do the required computations to adjust model parameters for each batch.
        Should use the provided interaction_dataset to do the training procedure.
        Must return the loss obtained on the current batch."""
        pass

    def predict(self, user_id, item_id, skip_errors=False, **kwds):
        """Performs a prediction using the provided user_id and item_id.

        Args:
            user_id: An integer representing the raw user id.
            item_id: An integer representing the raw item id.
            skip_errors: A boolean that controls if errors should be avoided or if they should be be thrown.
                Default: False. An example would be calling predict(None, None): If skip_errors is True,
                then it would return None; else it would throw an error.

        Returns:
            A float value representing the predicted interaction for the provided item, user pair. Or None, if
            an error occurs and skip_errors = True.
        """
        assert self.fitted is True, 'The model requires to be fitted before being able to make predictions.'
        assert skip_errors or self.interaction_dataset.user_to_uid(user_id) is not None, f'User {user_id} was not found.'
        assert skip_errors or self.interaction_dataset.item_to_iid(item_id) is not None, f'Item {item_id} was not found.'

        prediction = None

        try:
            uid = self.interaction_dataset.user_to_uid(user_id)
            iid = self.interaction_dataset.item_to_iid(item_id)
            prediction = self._predict(uid, iid, **kwds)
            if prediction is None:
                raise Exception(f'Failed to predict(user_id={user_id}, item_id={item_id}): None was returned.')
        except Exception as e:
            if not skip_errors: raise e

        return prediction

    @abstractmethod
    def _predict(self, uid, iid, **kwds):
        """Abstract method that should return a float value representing the predicted interaction for the
        provided item, user pair. uid and iid are internal ids."""
        pass

    def recommend(self, user_id, n=None, novelty=True, **kwds):
        """Computes a recommendation list for the given user and with the requested characteristics.

        Args:
            user_id: A string or integer representing the user id.
            n: An integer representing the number of recommended items.
            novelty: An optional boolean indicating if we only novelty recommendations or not. Default: True.
            interaction_threshold: Optional float value that represents the similarity value required to consider
                an item to be a useful recommendation Default: self.interaction_threshold.

        Returns:
            A list containing recommendations in the form of (similarity, item) tuples.
        """
        assert self.fitted is True, 'The model requires to be fitted before being able to make predictions.'
        assert self.interaction_dataset.user_to_uid(user_id) is not None, f'User {user_id} was not found.'

        if n is None: n = self.n_items

        threshold = kwds.get('interaction_threshold', self.interaction_threshold)
        uid = self.interaction_dataset.user_to_uid(user_id)
        recs = self._recommend(uid, n, novelty, threshold)
        return [(r, self.interaction_dataset.iid_to_item(iid)) for r, iid in recs]

    def _recommend(self, uid, n, novelty, threshold):
        """Returns a list containing recommendations in the form of (similarity, item) tuples. uid is an internal id."""
        iids = range(0, self.n_items)
        ranked_items = self._rank(uid, iids, n, novelty)
        return list(filter(lambda x: x[0] >= threshold, ranked_items))

    def rank(self, user_id, item_ids, novelty=True, skip_invalid_items=True, **kwds):
        """Ranks the provided item list for the given user and with the requested characteristics.

        Args:
            user_id: A string or integer representing the user id.
            item_ids: A list of strings or integers representing the ids of the items to rank.
            novelty: Optional boolean indicating if we only novelty recommendations or not. Default: True.
            n: Optional integer representing the number of best items to return. Default: len(item_ids).
            skip_invalid_items: Optional boolean indicating if invalid items should be skipped.
                If set to False, will throw an exception when one is found. Default: True.

        Returns:
            A ranked item list in the form of (similarity, item) tuples.
        """
        assert self.fitted is True, 'The model requires to be fitted before being able to make predictions.'
        assert self.interaction_dataset.user_to_uid(user_id) is not None, f'User {user_id} was not found.'

        uid = self.interaction_dataset.user_to_uid(user_id)
        iids = []
        for item_id in item_ids:
            iid = self.interaction_dataset.item_to_iid(item_id)
            if iid is not None:
                iids.append(iid)
            elif not skip_invalid_items:
                raise Exception(f'Item {item_id} was not found.')

        n = kwds.get('n', len(iids))
        assert n <= len(iids), \
            f'The number of best items to return must be <= len(item_ids) (current value is {n} > {len(iids)})'

        ranked_list = self._rank(uid, iids, n, novelty)
        return [(r, self.interaction_dataset.iid_to_item(iid)) for r, iid in ranked_list]

    def _rank(self, uid, iids, n, novelty):
        """Returns a ranked item list in the form of (similarity, item) tuples. uid and iids are internal ids."""
        if novelty:
            rated_items = self.interaction_dataset.select(f'uid == {uid}').values_list('iid', to_list=True)
            iids = set(iids).difference(set(rated_items))

        pred_list = filter(lambda x: x[0] is not None, [(self._predict(uid, iid), iid) for iid in iids])
        return nlargest(n, pred_list)

    def _standardize_value(self, value):
        """Standardizes a value in the [self.min_interaction, self.max_interaction] range, to the [0, 1] range."""
        return (value - self.min_interaction) / (self.max_interaction - self.min_interaction)

    def _rescale_value(self, value):
        """Rescales a standardized value in the [0, 1] range, to the [self.min_interaction, self.max_interaction] range."""
        return self.min_interaction + (self.max_interaction - self.min_interaction) * value

    def _log(self, msg):
        if not self.verbose: return
        print('[{name}] {msg}'.format(name=self.__class__.__name__, msg=msg))

    def save(self, save_path):  # todo: need interactionsDataset when saving?
        """Save/export the current model.

        Args:
            save_path: A string that represents the path in which the model will be saved.

        Returns:
            None.
        """
        dump(self, save_path)

    @staticmethod
    def load(load_path):  # todo: need interactionsDataset when loading?
        """Load/import a saved/exported model.

        Args:
            load_path: A string that represents the path to the saved/exported model.

        Returns:
            Recommender model.
        """
        return load(load_path)
