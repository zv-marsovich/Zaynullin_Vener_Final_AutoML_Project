import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from src.train_optuna import rmse_scorer


def test_rmse_scorer():
    """Тест функции RMSE"""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

    rmse = rmse_scorer(y_true, y_pred)
    assert rmse > 0
    assert isinstance(rmse, float)


def test_model_training_small():
    """Тест обучения на маленьких данных"""
    from sklearn.linear_model import LinearRegression

    X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    assert rmse < 10


def test_mlflow_logging():
    """Тест логирования MLflow"""
    import mlflow

    mlflow.set_experiment("test_experiment")
    with mlflow.start_run():
        mlflow.log_metric("test_metric", 0.5)
        mlflow.log_param("test_param", "value")

    assert mlflow.active_run() is None