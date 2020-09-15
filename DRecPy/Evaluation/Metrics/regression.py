from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from abc import abstractmethod
from .metric_abc import MetricABC


class PredictiveMetricABC(MetricABC):
    @abstractmethod
    def __call__(self, y_true, y_pred):
        """
        Args:
            y_true: A list containing the expected values.
            y_pred: A list containing the predicted values.

        Returns:
            The computed metric value.
        """
        pass


class MSE(PredictiveMetricABC):
    """Mean Squared Error."""
    def __call__(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)


class RMSE(PredictiveMetricABC):
    """Root Mean Squared Error."""
    def __call__(self, y_true, y_pred):
        return sqrt(mean_squared_error(y_true, y_pred))


class MAE(PredictiveMetricABC):
    """Mean absolute Error."""
    def __call__(self, y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)
