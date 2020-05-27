from math import sqrt
from sklearn.metrics import mean_squared_error


def mse(y_true, y_pred):
    """Mean Squared Error.

    Args:
        y_true: A list containing the expected values.
        y_pred: A list containing the predicted values.

    Returns:
        The computed Mean Squared Error value.
    """
    return mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    """Root Mean Squared Error.

    Args:
        y_true: A list containing the expected values.
        y_pred: A list containing the predicted values.

    Returns:
        The computed Root Mean Squared Error value.
    """
    return sqrt(mse(y_true, y_pred))
