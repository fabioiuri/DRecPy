from DRecPy.Evaluation.Metrics import mse
from DRecPy.Evaluation.Metrics import rmse


def test_mse_0():
    assert mse([1, 2, 3, 4], [1, 2, 3, 4]) == 0


def test_mse_1():
    assert mse([4, 3, 2, 1], [1, 2, 3, 4]) == 5


def test_mse_2():
    assert round(mse([5, 10, 2, -1, -20, 6], [9, 90, 24, 2, 0, -15]), 2) == 1291.67


def test_mse_3():
    assert mse([0, 0], [1, 1]) == 1


def test_mse_4():
    assert mse([0], [0]) == 0


def test_rmse_0():
    assert rmse([1, 2, 3, 4], [1, 2, 3, 4]) == 0


def test_rmse_1():
    assert round(rmse([4, 3, 2, 1], [1, 2, 3, 4]), 2) == 2.24


def test_rmse_2():
    assert round(rmse([5, 10, 2, -1, -20, 6], [9, 90, 24, 2, 0, -15]), 2) == 35.94


def test_rmse_3():
    assert rmse([0, 0], [1, 1]) == 1


def test_rmse_4():
    assert rmse([0], [0]) == 0
