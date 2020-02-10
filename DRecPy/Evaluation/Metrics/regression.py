from math import sqrt
from sklearn.metrics import mean_squared_error


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return sqrt(mse(y_true, y_pred))
