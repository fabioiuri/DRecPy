from .early_stopping_rule_abc import EarlyStoppingRuleABC


class MaxValidationValueRule(EarlyStoppingRuleABC):
    """MaxValidationValueRule early stopping.

    This early stopping rule consists in checking the epoch that achieves the maximum validation value, never stopping
    the training procedure.

    Args:
        validation_metric: The name of the required metric to compute the max. validation value rule.
    """
    def __init__(self, validation_metric, **kwds):
        super(MaxValidationValueRule, self).__init__(required_validation_metrics=[validation_metric])

        self.validation_metric = validation_metric

    def _compute_best_epoch(self, epoch_losses, epoch_validation_results, called_epochs_validation_results, **kwds):
        best_epoch_idx = 0
        best_epoch_val_res = epoch_validation_results[self.validation_metric][0]

        for epoch_idx, epoch_val_res in enumerate(epoch_validation_results[self.validation_metric]):
            if epoch_val_res > best_epoch_val_res:
                best_epoch_val_res = epoch_val_res
                best_epoch_idx = epoch_idx

        return called_epochs_validation_results[best_epoch_idx]

    def stop_training(self, current_epoch, best_computed_epoch, target_epoch, **kwds):
        return False
