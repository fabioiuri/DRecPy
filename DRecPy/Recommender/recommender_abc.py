from abc import ABC
from abc import abstractmethod
from joblib import load
from joblib import dump
from tqdm import tqdm
from DRecPy.Evaluation import LossTracker
from heapq import nlargest
import random
import tensorflow as tf
import logging
from datetime import datetime
import copy

from DRecPy.Recommender.EarlyStopping.early_stopping_rule_abc import InvalidEpochValidationResultsException

tf.config.set_soft_device_placement(True)  # automatically choose an existing and supported device to run (GPU, CPU)


class RecommenderABC(ABC):
    """Base recommender abstract class.

    This class implements the skeleton methods required for building a recommender.
    It provides id-abstraction (handles conversion between raw to internal ids,
    auto identifier validation (if a given user/item is known or not), automatic progress logging, weight updates,
    tracking loss per epoch and support for other features such as epoch callbacks and early stopping. It has a
    structure that allows it to be fully extensible, whilst promoting model specific behavior for improved flexibility.
    All private methods are called with internal ids only, and all public methods must be called with raw ids only.

    The following methods are still required to be implemented: _pre_fit(), _sample_batch(), _predict_batch(),
    _compute_batch_loss() and _predict().
    If there are no trainable variables set during the _pre_fit(), batch training is skipped (useful for non-deep
    learning models). Trainable variables can be registered via _register_trainable() or _register_trainables().

    If the provided trainable variables are of type tf.keras.models.Model or tf.keras.layers.Layer, then the
    regularizers added when instantiating these variables (e.g. via kernel_regularizer attribute of a layer) will be
    used to automatically compute the regularization loss. To add a custom regularization loss
    (or if the implemented model uses tf.Variables), implement the self._compute_reg_loss().

    Optionally, these methods can be overridden: _rank() and _recommend().

    Args:
        verbose: Optional boolean indicating if the recommender should print progress logs or not.
            Default: True.
        log_file: Optional boolean indicating if a file containing all produced logs should be created or not.
            It will be created on the current directory, following the pattern: drecpy__DATE_TIME_RECNAME.log.
            Default: False.
        interaction_threshold: An optional integer that is used as the boundary interaction value between positive and
            negative interaction pairs. All values above or equal interaction_threshold are considered positive, and
            all values bellow are considered negative. Default: 0.001.
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
        self.trainable_vars = []
        self.trainable_weights = []
        self.trainable_layers = []
        self.trainable_models = []
        self.epoch_weights = []
        self.optimizer = None

        self._loss_tracker = None
        self._rng = random.Random(self.seed)
        tf.random.set_seed(self.seed)

        log_formatter = logging.Formatter('[%(asctime)s] (%(levelname)s) %(name)s: %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(log_formatter)
        ch.setLevel(logging.INFO)
        self._logger = logging.getLogger(f'{self.__class__.__name__}_CLOGGER')
        self._logger.propagate = False
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()
        self._logger.addHandler(ch)
        self._file_logger = None

        if kwds.get('log_file', False):
            fh = logging.FileHandler(f'drecpy_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{self.__class__.__name__}.log')
            fh.setLevel(logging.INFO)
            fh.setFormatter(log_formatter)
            self._file_logger = logging.getLogger(f'{self.__class__.__name__}_FLOGGER')
            self._file_logger.propagate = False
            self._file_logger.setLevel(logging.INFO)
            self._file_logger.addHandler(fh)

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
                is set to True. Default: None
            epoch_callback_freq: Optional integer representing the frequency in which the epoch_callback_fn is called.
                If epoch_callback_fn is not defined, this parameter is ignored. Default: 5 (called every 5 epochs).
            early_stopping_rule: Optional instance of EarlyStoppingRuleABC that will be used to compute the early
                stopping epoch and according to how the rule works, it might stop the training before achieving the
                total number of epochs. Since this uses epoch callbacks, the rule will only be used if an
                epoch_callback_fn is defined. Default: None
            early_stopping_freq: Optional integer representing the frequency in which the early_stopping_rule is
                computed. If early_stopping_rule is not defined, this parameter is ignored. Default: 5 (called every 5
                epochs).
            copy_dataset: Optional boolean indicating weather a copy of the given dataset should be made.
                If set to False, the given dataset instance is used. Default: False.
            optimizer: Optional instance of a tf/keras optimizer that will be used even if there's a model specific
                optimizer. Default: Adam optimizer with the learning rate set with the value provided in the
                learning_rate argument; if there's a model specific optimizer (set during the model's _pre_fit), this
                default optimizer will not be used.

        Returns:
            None.
        """
        self.interaction_dataset = interaction_dataset
        if copy_dataset:
            self._info('Cloning new dataset instance...')
            self.interaction_dataset = interaction_dataset.__copy__()
        self.interaction_dataset.assign_internal_ids()

        self.min_interaction = self.interaction_dataset.min('interaction')
        if self.min_interaction == 1: self.min_interaction = 0
        self.max_interaction = self.interaction_dataset.max('interaction')

        self.n_users = self.interaction_dataset.count_unique('uid')
        self.n_items = self.interaction_dataset.count_unique('iid')
        self.n_rows = len(self.interaction_dataset)

        self._loss_tracker = LossTracker()

        self._log_initial_info()

        self._info('Creating auxiliary structures...')
        self._register_optimizer(tf.keras.optimizers.Adam(learning_rate=learning_rate))  # default optimizer
        self._pre_fit(learning_rate, neg_ratio, reg_rate, **kwds)
        if kwds.get('optimizer', None) is not None:  # allow forcing custom optimizer
            self._register_optimizer(kwds.get('optimizer'))
        self.fitted = True  # should be able to make predictions after pre fit

        if len(self.trainable_weights) == 0 and len(self.trainable_layers) == 0 and len(self.trainable_models) == 0:
            self._info('No trainable vars found: skipping further model training. If this is non-intentional, please '
                      'use self._register_trainable or self._register_trainables to register variables that are '
                      'subject to weight updates.')
            return

        self._info(f'Number of registered trainable variables: '
                   f'{len(self.trainable_weights) + len(self.trainable_layers) + len(self.trainable_models)}')

        progress_desc = ''

        epoch_callback_fn = kwds.get('epoch_callback_fn', None)
        epoch_callback_ret, epoch_callback_res_registered = None, True
        epoch_callback_freq = kwds.get('epoch_callback_freq', 5)

        early_stopping_rule = kwds.get('early_stopping_rule', None)
        early_stopping_freq = kwds.get('early_stopping_freq', 5)
        early_stopping_best_epoch = None

        if self.verbose and epoch_callback_fn is not None:
            epoch_callback_ret = epoch_callback_fn(self)
            assert type(epoch_callback_ret) is dict, \
                f'The return type of the epoch_callback_fn should be dict, but found {type(epoch_callback_ret)}'

            for metric in epoch_callback_ret:
                self._loss_tracker.add_epoch_callback_result(metric, epoch_callback_ret[metric], 0)

        _iter = range(1, epochs+1)
        if self.verbose:
            _iter = tqdm(range(1, epochs+1), total=epochs, desc='Fitting model...', position=0, leave=True)
        for e in _iter:
            batch_samples = self._sample_batch(batch_size, **kwds)
            with tf.GradientTape() as tape:
                predictions, desired_values = self._predict_batch(batch_samples, **kwds)

                trainable_weights = self.trainable_weights + \
                                    [layer.trainable_weights for layer in self.trainable_layers] + \
                                    [model.trainable_weights for model in self.trainable_models]
                for trainable_weight in trainable_weights:
                    tape.watch(trainable_weight)

                loss = self._compute_batch_loss(predictions, desired_values, **kwds) + \
                    self._compute_reg_loss(reg_rate, len(batch_samples), self.trainable_models, self.trainable_layers, self.trainable_weights)

            gradients = tape.gradient(loss, trainable_weights)
            self._update_weights(gradients, trainable_weights)
            self._store_epoch_weights()

            if self.verbose or early_stopping_rule is not None:
                # only need to compute epoch callbacks when verbose is active or with early stopping
                self._loss_tracker.add_epoch_loss(loss)
                if epoch_callback_fn is not None and e % epoch_callback_freq == 0:
                    epoch_callback_res_registered = False
                    epoch_callback_ret = epoch_callback_fn(self)
                    assert isinstance(epoch_callback_ret, dict), \
                        f'The return type of the epoch_callback_fn should be dict, but found {type(epoch_callback_ret)}'

                progress_desc = f'Fitting model... Epoch {e} Loss: {loss:.4f}'
                if epoch_callback_ret is not None:
                    for metric in epoch_callback_ret:
                        progress_desc += f' | {metric}: {epoch_callback_ret[metric]}'
                        if not epoch_callback_res_registered:
                            self._loss_tracker.add_epoch_callback_result(metric, epoch_callback_ret[metric], e)
                    epoch_callback_res_registered = True

            if early_stopping_rule is not None and e % early_stopping_freq == 0:
                try:
                    early_stopping_best_epoch = early_stopping_rule.compute(self._loss_tracker.epoch_losses,
                                                                            self._loss_tracker.epoch_callback_results,
                                                                            self._loss_tracker.called_epochs)
                    if early_stopping_rule.stop_training(e, early_stopping_best_epoch, epochs):
                        break
                except InvalidEpochValidationResultsException as e:
                    self._warn(f'Failed to compute early stopping rule {early_stopping_rule.__class__.__name__}: {e}')

            if early_stopping_best_epoch is not None:
                progress_desc += f' | {early_stopping_rule.__class__.__name__} best epoch: {early_stopping_best_epoch}'

            if self.verbose:
                _iter.set_description(progress_desc)
            self._info(progress_desc, log_console=False)

        if early_stopping_rule is not None and e % early_stopping_freq != 0:  # not already computed on last epoch
            try:
                early_stopping_best_epoch = early_stopping_rule.compute(self._loss_tracker.epoch_losses,
                                                                        self._loss_tracker.epoch_callback_results,
                                                                        self._loss_tracker.called_epochs)
                progress_desc += f' | {early_stopping_rule.__class__.__name__} best epoch: {early_stopping_best_epoch}'
                if self.verbose:
                    _iter.set_description(progress_desc)
                self._info(progress_desc, log_console=False)
            except InvalidEpochValidationResultsException as e:
                self._warn(f'Failed to compute early stopping rule {early_stopping_rule.__class__.__name__}: {e}')

        if early_stopping_best_epoch is not None and early_stopping_best_epoch != epochs:
            self._info(f'Reverting network weights to epoch {early_stopping_best_epoch} due to the evaluation of the '
                       f'early stopping rule {early_stopping_rule.__class__.__name__}.')
            self._revert_weights(early_stopping_best_epoch)

        if self.verbose:
            self._loss_tracker.display_graph(
                model_name=self.__class__.__name__,
                stopping_epoch=early_stopping_best_epoch if early_stopping_best_epoch != epochs else None
            )

        self._info('Model fitted.')

    def _register_trainable(self, variable):
        if variable is None:
            raise Exception('Cannot register None as a trainable variable.')

        if isinstance(variable, tf.keras.models.Model):
            self.trainable_models.append(variable)
        elif isinstance(variable, tf.keras.layers.Layer):
            self.trainable_layers.append(variable)
        elif tf.is_tensor(variable):
            self.trainable_weights.append(variable)
        else:
            raise Exception(f'Invalid trainable variable {variable}. The supported types are: tf.keras.models.Model, '
                            f'tf.keras.layers.Layer and tf.Variable.')

    def _register_trainables(self, variables):
        for variable in variables:
            self._register_trainable(variable)

    def _register_optimizer(self, optimizer):
        self.optimizer = optimizer

    @abstractmethod
    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        """Abstract method that should setup all the required structures for fitting the model.
        Should use the provided interaction_dataset to do the training procedure."""
        pass

    @abstractmethod
    def _sample_batch(self, batch_size, **kwds):
        """Abstract method that should sample batch_size training data points, which are then passed to the
        _predict_batch method. The return format is not rigid in order to improve flexibility, but should contain
        at least the following information: inputs and desired outputs."""
        pass

    @abstractmethod
    def _predict_batch(self, batch_samples, **kwds):
        """Abstract method that should compute predictions for each of the data points contained on the batch_samples
        argument. The return format is fixed, and should be a tuple of predictions (list) and desired values (list),
        in this exact order."""
        pass

    @abstractmethod
    def _compute_batch_loss(self, predictions, desired_values, **kwds):
        """Abstract method that should compute the batch (prediction) loss for the given predictions and desired_values.
        This loss value must be differentiable with respect to the model's parameters (variables that were registered
        through the _register_trainable or _register_trainables)."""
        pass

    def _compute_reg_loss(self, reg_rate, batch_size, trainable_models, trainable_layers, trainable_weights, **kwds):
        """Method that computes the model's regularization loss, using the provided regularization
        rate and taking into account that the updates to the weights are made for every batch_size iterations.

        The default implementation is only useful when the registered trainable variables are of type
        tf.keras.models.Model or tf.keras.layers.Layer, in which case the registered regularizers (e.g. via
        kernel_regularizer attribute of a layer) are added to the loss of the recommender for each batch.

        If the registered variables are of type tf.Variable, then the trainable_layers and trainable_models parameters
        will be empty lists, and the trainable_weights parameter will contain these tf.Variables that are updated for
        each epoch."""
        return sum([tf.math.add_n(trainable.losses) if len(trainable.losses) > 0 else 0
                    for trainable in trainable_layers + trainable_models])

    def _update_weights(self, gradients, trainable_weights):
        for gradient, trainable_var in zip(gradients, trainable_weights):
            if tf.is_tensor(gradient):
                gradient = [gradient]
            if tf.is_tensor(trainable_var):
                trainable_var = [trainable_var]
            self.optimizer.apply_gradients(zip(gradient, trainable_var))

    def _store_epoch_weights(self):
        self.epoch_weights.append({
            'trainable_weights': copy.deepcopy(self.trainable_weights),
            'trainable_layers': [layer.trainable_weights for layer in self.trainable_layers],
            'trainable_models': [model.trainable_weights for model in self.trainable_models]
        })

    def _revert_weights(self, epoch):
        target_weights = self.epoch_weights[epoch-1]  # convert to 0-based idx
        for model, weights in zip(self.trainable_models, target_weights['trainable_models']):
            model.set_weights(weights)
        for layer, weights in zip(self.trainable_layers, target_weights['trainable_layers']):
            layer.set_weights(weights)
        for variable, weights in zip(self.trainable_weights, target_weights['trainable_weights']):
            variable.assign(weights)

        self._info(f'Network weights reverted from epoch {len(self.epoch_weights)} to epoch {epoch}.')

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

    def _log_initial_info(self):
        self._info(f'Max. interaction value: {self.max_interaction}')
        self._info(f'Min. interaction value: {self.min_interaction}')
        self._info(f'Interaction threshold value: {self.interaction_threshold}')
        self._info(f'Number of unique users: {self.n_users}')
        self._info(f'Number of unique items: {self.n_items}')
        self._info(f'Number of training points: {self.n_rows}')
        matrix_size = self.n_users * self.n_items
        sparsity = round(100 * (1 - (self.n_rows / matrix_size)), 4)
        self._info(f'Sparsity level: approx. {sparsity}%')

    def _info(self, msg, log_console=True, log_file=True):
        if not self.verbose: return
        if log_console:
            self._logger.info(msg)
        if self._file_logger and log_file:
            self._file_logger.info(msg)

    def _warn(self, msg, log_console=True, log_file=True):
        if not self.verbose: return
        if log_console:
            self._logger.warning(msg)
        if self._file_logger and log_file:
            self._file_logger.warning(msg)

    def _error(self, msg, log_console=True, log_file=True):
        if not self.verbose: return
        if log_console:
            self._logger.error(msg)
        if self._file_logger and log_file:
            self._file_logger.error(msg)

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
