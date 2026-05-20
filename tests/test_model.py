import numpy as np

from src.train_optuna import rmse_scorer


def test_rmse_scorer():
    """Тест функции RMSE"""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

    rmse = rmse_scorer(y_true, y_pred)
    assert rmse > 0
    assert isinstance(rmse, float)


def test_rmse_perfect_prediction():
    """Тест RMSE при идеальном предсказании"""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    rmse = rmse_scorer(y_true, y_pred)
    assert rmse == 0


def test_rmse_negative():
    """Тест RMSE с отрицательными значениями"""
    y_true = np.array([-5, -4, -3, -2, -1])
    y_pred = np.array([-4.9, -4.1, -2.9, -2.1, -0.9])

    rmse = rmse_scorer(y_true, y_pred)
    assert rmse > 0