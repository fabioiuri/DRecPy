from abc import ABC
from abc import abstractmethod


class EarlyStoppingRuleABC(ABC):
    """Base early stopping rule class.

    This base class provides the structure for all early stopping rules (which need to be the same so that
    the recommender base class workflow doesn't break), as well as some utility checks and methods.

    Args:
        required_validation_metrics: A list containing the names of the required metrics to compute this rule.
    """
    def __init__(self, required_validation_metrics, **kwds):
        self.required_validation_metrics = [] if required_validation_metrics is None else required_validation_metrics

        if len(self.required_validation_metrics) == 0:
            raise InvalidRequiredValidationMetricsException('The required validation metrics parameter must be a '
                                                            'non-empty list with the names of the required validation metrics.')

    def compute(self, epoch_losses, epoch_validation_results, called_epochs_validation_results, **kwds):
        """Computes the best epoch taking into account the current early stopping rule instance and the provided epoch
        losses and epoch validation results.

        Args:
            epoch_losses: A list of size E, where E is the total number of epochs, containing the loss values the
                model has shown for each epoch.
            epoch_validation_results: A dict mapping metric names to a list of validation values obtained on those metrics.
            called_epochs_validation_results: A list containing as many elements as each list in the epoch_validation_results
                values, where each element represent the epoch on which the i-th validation result was computed.

        Returns:
            An integer representing the best epoch according to the current early stopping rule.
        """
        if not isinstance(epoch_validation_results, dict):
            raise InvalidEpochValidationResultsException(f'Epoch callback results must be a dict. Found: {epoch_validation_results}.')

        if len(epoch_validation_results.keys()) == 0:
            raise InvalidEpochValidationResultsException(f'Epoch callback results must be a non-empty dict.')

        if any(map(lambda callback_result: not isinstance(callback_result, list), epoch_validation_results.values())):
            raise InvalidEpochValidationResultsException(f'All epoch callback results should map to a list of values. '
                                                        f'Found: {epoch_validation_results.values()}.')

        if any(map(lambda val_metric: "@" in val_metric, self.required_validation_metrics)):
            # if required validation metrics require @ metrics, don't trim them
            validation_metrics = {metric: metric for metric in epoch_validation_results.keys()}
        else:
            validation_metrics = {metric.split("@")[0]: metric for metric in epoch_validation_results.keys()}

        if not set(self.required_validation_metrics).issubset(set(validation_metrics.keys())):
            raise InvalidEpochValidationResultsException('No matching epoch callback metric with the required validation'
                                                        f' metrics. Expected: {self.required_validation_metrics}, '
                                                        f'found: {validation_metrics.keys()}.')

        translated_epoch_validation_results = {val_metric: epoch_validation_results[validation_metrics[val_metric]]
                                              for val_metric in validation_metrics}

        if not all(map(lambda required_val_metric: len(translated_epoch_validation_results[required_val_metric]) > 0,
                       self.required_validation_metrics)):
            raise InvalidEpochValidationResultsException(f'All epoch callback results should map to a non-empty list of '
                                                         f'values. Found: {epoch_validation_results.values()}.')

        return self._compute_best_epoch(epoch_losses, translated_epoch_validation_results,
                                        called_epochs_validation_results)

    @abstractmethod
    def _compute_best_epoch(self, epoch_losses, epoch_validation_results, called_epochs_validation_results, **kwds):
        """Should return an integer representing the best epoch according to the current early stopping rule."""
        pass

    @abstractmethod
    def stop_training(self, current_epoch, best_computed_epoch, target_epoch, **kwds):
        """Computes whether the training process should stop at the current epoch, or if it should continue.

        Args:
            current_epoch: An integer representing the current training epoch.
            best_computed_epoch: An integer representing the best compute epoch by the current early stopping rule.
            target_epoch: An integer representing the target number of epochs for the training process.

        Returns:
            A boolean indicating whether to stop the training process or continue.
        """
        pass


class InvalidRequiredValidationMetricsException(Exception):
    pass


class InvalidEpochValidationResultsException(Exception):
    pass
