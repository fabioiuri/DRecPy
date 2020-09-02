from DRecPy.Evaluation.Metrics import MSE
from DRecPy.Evaluation.Metrics import RMSE


def test_mse_0():
    assert MSE()([1, 2, 3, 4], [1, 2, 3, 4]) == 0


def test_mse_1():
    assert MSE()([4, 3, 2, 1], [1, 2, 3, 4]) == 5


def test_mse_2():
    assert round(MSE()([5, 10, 2, -1, -20, 6], [9, 90, 24, 2, 0, -15]), 2) == 1291.67


def test_mse_3():
    assert MSE()([0, 0], [1, 1]) == 1


def test_mse_4():
    assert MSE()([0], [0]) == 0


def test_rmse_0():
    assert RMSE()([1, 2, 3, 4], [1, 2, 3, 4]) == 0


def test_rmse_1():
    assert round(RMSE()([4, 3, 2, 1], [1, 2, 3, 4]), 2) == 2.24


def test_rmse_2():
    assert round(RMSE()([5, 10, 2, -1, -20, 6], [9, 90, 24, 2, 0, -15]), 2) == 35.94


def test_rmse_3():
    assert RMSE()([0, 0], [1, 1]) == 1


def test_rmse_4():
    assert RMSE()([0], [0]) == 0
