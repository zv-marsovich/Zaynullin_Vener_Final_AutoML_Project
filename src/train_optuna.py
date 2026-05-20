import time
import os
import psutil
from datetime import datetime

import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.sklearn
import joblib

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
from catboost import cv as catboost_cv
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

from .preprocessing import (
    run_preprocessing, get_features_target, get_columns_types,
    create_preprocessor, split_data
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
os.makedirs('mlruns', exist_ok=True)
os.makedirs('models', exist_ok=True)
mlflow.set_tracking_uri('file://' + os.path.join(PROJECT_ROOT, 'mlruns'))


def rmse_scorer(y_true, y_pred):
    """Расчет RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def log_system_metrics():
    """Логирование использования системы в MLflow"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent

    mlflow.log_metric("system_cpu_percent", cpu_percent)
    mlflow.log_metric("system_memory_percent", memory_percent)
    mlflow.log_metric("system_disk_usage", disk_usage)

    return cpu_percent, memory_percent


def objective_xgb(trial, X_train, y_train, preprocessor):
    """Целевая функция Optuna для XGBoost"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'objective': 'reg:squarederror',
        'verbosity': 0
    }

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(**params))
    ])

    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=3, scoring='neg_root_mean_squared_error',
                             n_jobs=-1)

    return -scores.mean()


def objective_catboost(trial, X_train, y_train, cat_cols):
    """Целевая функция Optuna для CatBoost"""
    params = {
        'iterations': trial.suggest_int('iterations', 500, 1500, step=100),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_seed': 42,
        'verbose': 0,
        'loss_function': 'RMSE'
    }
    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=cat_cols
    )
    cv_data = catboost_cv(
        params=params,
        pool=train_pool,
        fold_count=3,
        shuffle=True,
        verbose=False
    )

    return cv_data['test-RMSE-mean'].values[-1]


def objective_lgbm(trial, X_train, y_train, preprocessor, cat_cols):
    """Целевая функция Optuna для LightGBM"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbose': -1
    }

    X_train_lgb = X_train.copy()
    for col in cat_cols:
        if col in X_train_lgb.columns:
            X_train_lgb[col] = X_train_lgb[col].astype('category')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LGBMRegressor(**params))
    ])

    scores = cross_val_score(pipeline, X_train_lgb, y_train,
                             cv=3, scoring='neg_root_mean_squared_error',
                             n_jobs=-1)

    return -scores.mean()


def train_with_optuna(n_trials=10):
    """Обучение всех моделей с Optuna"""
    print("=" * 60)
    print("AutoML с Optuna (байесовская оптимизация)")
    print(f"Время старта: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"CPU ядер: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024 ** 3):.1f} GB")
    print("=" * 60)

    print("\n 1. Загрузка и предобработка данных...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, '..', 'data', 'autos.csv')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Датасет не найден: {filepath}")

    df = run_preprocessing(filepath)
    print(f"Данные загружены: {df.shape[0]} строк, {df.shape[1]} колонок")

    features, target = get_features_target(df)
    global cat_cols, num_cols
    cat_cols, num_cols = get_columns_types(features)
    print(f"Категориальные признаки: {len(cat_cols)}")
    print(f"Числовые признаки: {len(num_cols)}")

    X_train, X_test, y_train, y_test = split_data(features, target, test_size=0.2, random_state=42)
    print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    preprocessor = create_preprocessor(cat_cols, num_cols)
    preprocessor.fit(X_train)

    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    results = {}
    best_models = {}

    mlflow.set_experiment("car_price_prediction_optuna")

    with mlflow.start_run(run_name=f"AutoML_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params({
            "n_trials": n_trials,
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
            "n_categorical_features": len(cat_cols),
            "n_numerical_features": len(num_cols)
        })
        log_system_metrics()

        # 1. LinearRegression
        print("\n Обучение LinearRegression (benchmark)...")
        start_time = time.time()
        lr_pipeline = Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])
        lr_scores = cross_val_score(lr_pipeline, X_train, y_train,
                                    cv=3, scoring='neg_root_mean_squared_error')
        lr_rmse = -lr_scores.mean()

        lr_pipeline.fit(X_train, y_train)
        lr_time = time.time() - start_time

        y_pred_lr = lr_pipeline.predict(X_test)
        lr_test_rmse = rmse_scorer(y_test, y_pred_lr)
        lr_test_mae = mean_absolute_error(y_test, y_pred_lr)
        lr_test_r2 = r2_score(y_test, y_pred_lr)

        results['LinearRegression'] = {
            'rmse_cv': lr_rmse,
            'rmse_test': lr_test_rmse,
            'mae_test': lr_test_mae,
            'r2_test': lr_test_r2,
            'train_time': lr_time,
            'best_params': None
        }
        best_models['LinearRegression'] = lr_pipeline

        print(f"CV RMSE: {lr_rmse:.3f}")
        print(f"Test RMSE: {lr_test_rmse:.3f}, MAE: {lr_test_mae:.3f}, R2: {lr_test_r2:.3f}")
        print(f"Time: {lr_time:.2f}s")

        mlflow.log_metrics(
            {
                "LinearRegression_cv_rmse": lr_rmse,
                "LinearRegression_test_rmse": lr_test_rmse,
                "LinearRegression_test_mae": lr_test_mae,
                "LinearRegression_test_r2": lr_test_r2,
                "LinearRegression_train_time": lr_time
            }
        )

        # 2. CatBoost с Optuna
        print("\n Оптимизация CatBoost с Optuna...")
        start_time = time.time()
        log_system_metrics()

        X_train_cat = X_train.copy()
        X_test_cat = X_test.copy()

        study_cat = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5)
        )

        study_cat.optimize(
            lambda trial: objective_catboost(trial, X_train_cat, y_train, cat_cols),
            n_trials=n_trials,
            n_jobs=1,
            show_progress_bar=True
        )

        cat_time = time.time() - start_time
        log_system_metrics()

        best_cat_params = study_cat.best_params.copy()
        best_cat_model = CatBoostRegressor(
            **best_cat_params,
            cat_features=cat_cols,
            random_seed=42,
            verbose=0
        )
        best_cat_model.fit(X_train_cat, y_train)

        y_pred_cat = best_cat_model.predict(X_test_cat)
        cat_test_rmse = rmse_scorer(y_test, y_pred_cat)
        cat_test_mae = mean_absolute_error(y_test, y_pred_cat)
        cat_test_r2 = r2_score(y_test, y_pred_cat)

        results['CatBoost'] = {
            'rmse_cv': study_cat.best_value,
            'rmse_test': cat_test_rmse,
            'mae_test': cat_test_mae,
            'r2_test': cat_test_r2,
            'train_time': cat_time,
            'best_params': best_cat_params,
            'n_trials': n_trials
        }
        best_models['CatBoost'] = best_cat_model

        print(f"\n Лучшие параметры: {best_cat_params}")
        print(f"CV RMSE: {study_cat.best_value:.3f}")
        print(f"Test RMSE: {cat_test_rmse:.3f}, MAE: {cat_test_mae:.3f}, R2: {cat_test_r2:.3f}")
        print(f"Time: {cat_time:.2f}s")

        log_system_metrics()
        mlflow.log_metrics(
            {
                "CatBoost_cv_rmse": study_cat.best_value,
                "CatBoost_test_rmse": cat_test_rmse,
                "CatBoost_test_mae": cat_test_mae,
                "CatBoost_test_r2": cat_test_r2,
                "CatBoost_train_time": cat_time
            }
        )

        mlflow.log_params({f"CatBoost_{k}": v for k, v in best_cat_params.items()})

        # 3. XGBoost с Optuna
        print("\n Оптимизация XGBoost с Optuna...")
        start_time = time.time()
        log_system_metrics()

        study_xgb = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )

        study_xgb.optimize(
            lambda trial: objective_xgb(trial, X_train, y_train, preprocessor),
            n_trials=n_trials,
            n_jobs=-1,
            show_progress_bar=True
        )

        xgb_time = time.time() - start_time
        log_system_metrics()

        best_xgb_params = study_xgb.best_params
        best_xgb_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBRegressor(**best_xgb_params, random_state=42, objective='reg:squarederror', verbosity=0))
        ])
        best_xgb_pipeline.fit(X_train, y_train)

        y_pred_xgb = best_xgb_pipeline.predict(X_test)
        xgb_test_rmse = rmse_scorer(y_test, y_pred_xgb)
        xgb_test_mae = mean_absolute_error(y_test, y_pred_xgb)
        xgb_test_r2 = r2_score(y_test, y_pred_xgb)

        results['XGBoost'] = {
            'rmse_cv': study_xgb.best_value,
            'rmse_test': xgb_test_rmse,
            'mae_test': xgb_test_mae,
            'r2_test': xgb_test_r2,
            'train_time': xgb_time,
            'best_params': best_xgb_params,
            'n_trials': n_trials
        }
        best_models['XGBoost'] = best_xgb_pipeline

        print(f"\n Лучшие параметры: {best_xgb_params}")
        print(f"CV RMSE: {study_xgb.best_value:.3f}")
        print(f"Test RMSE: {xgb_test_rmse:.3f}, MAE: {xgb_test_mae:.3f}, R2: {xgb_test_r2:.3f}")
        print(f"Time: {xgb_time:.2f}s")

        log_system_metrics()
        mlflow.log_metrics(
            {
                "XGBoost_cv_rmse": study_xgb.best_value,
                "XGBoost_test_rmse": xgb_test_rmse,
                "XGBoost_test_mae": xgb_test_mae,
                "XGBoost_test_r2": xgb_test_r2,
                "XGBoost_train_time": xgb_time
            }
        )
        mlflow.log_params({f"XGBoost_{k}": v for k, v in best_xgb_params.items()})

        # 4. LightGBM с Optuna
        print("\n Оптимизация LightGBM с Optuna...")
        start_time = time.time()
        log_system_metrics()

        study_lgb = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )

        study_lgb.optimize(
            lambda trial: objective_lgbm(trial, X_train, y_train, preprocessor, cat_cols),
            n_trials=n_trials,
            n_jobs=1,
            show_progress_bar=True
        )

        lgb_time = time.time() - start_time
        log_system_metrics()

        best_lgb_params = study_lgb.best_params

        X_train_lgb = X_train.copy()
        X_test_lgb = X_test.copy()
        for col in cat_cols:
            if col in X_train_lgb.columns:
                X_train_lgb[col] = X_train_lgb[col].astype('category')
                X_test_lgb[col] = X_test_lgb[col].astype('category')

        best_lgb_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', LGBMRegressor(**best_lgb_params, random_state=42, verbose=-1))
        ])
        best_lgb_pipeline.fit(X_train_lgb, y_train)

        y_pred_lgb = best_lgb_pipeline.predict(X_test_lgb)
        lgb_test_rmse = rmse_scorer(y_test, y_pred_lgb)
        lgb_test_mae = mean_absolute_error(y_test, y_pred_lgb)
        lgb_test_r2 = r2_score(y_test, y_pred_lgb)

        results['LightGBM'] = {
            'rmse_cv': study_lgb.best_value,
            'rmse_test': lgb_test_rmse,
            'mae_test': lgb_test_mae,
            'r2_test': lgb_test_r2,
            'train_time': lgb_time,
            'best_params': best_lgb_params,
            'n_trials': n_trials
        }
        best_models['LightGBM'] = best_lgb_pipeline

        print(f"\n Лучшие параметры: {best_lgb_params}")
        print(f"CV RMSE: {study_lgb.best_value:.3f}")
        print(f"Test RMSE: {lgb_test_rmse:.3f}, MAE: {lgb_test_mae:.3f}, R2: {lgb_test_r2:.3f}")
        print(f"Time: {lgb_time:.2f}s")

        log_system_metrics()
        mlflow.log_metrics(
            {
                "LightGBM_cv_rmse": study_lgb.best_value,
                "LightGBM_test_rmse": lgb_test_rmse,
                "LightGBM_test_mae": lgb_test_mae,
                "LightGBM_test_r2": lgb_test_r2,
                "LightGBM_train_time": lgb_time
            }
        )
        mlflow.log_params({f"LightGBM_{k}": v for k, v in best_lgb_params.items()})

        best_model_name = min(results, key=lambda x: results[x]['rmse_test'])
        best_model = best_models[best_model_name]
        final_rmse = results[best_model_name]['rmse_test']

        mlflow.log_metric("best_model_test_rmse", final_rmse)
        mlflow.log_param("best_model_name", best_model_name)

        log_system_metrics()

        joblib.dump(best_model, 'models/best_model_optuna.pkl')
        mlflow.log_artifact('models/best_model_optuna.pkl')
        mlflow.log_artifact('models/preprocessor.pkl')

        results_df = pd.DataFrame([
            {
                'Модель': name,
                'RMSE (CV)': round(metrics['rmse_cv'], 3),
                'RMSE (Test)': round(metrics['rmse_test'], 3),
                'MAE (Test)': round(metrics['mae_test'], 3),
                'R2 (Test)': round(metrics['r2_test'], 3),
                'Время (сек)': round(metrics['train_time'], 2)
            }
            for name, metrics in results.items()
        ])
        results_df.to_csv('models/results.csv', index=False)
        mlflow.log_artifact('models/results.csv')

        print("\n" + "=" * 60)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("=" * 60)
        print(results_df.to_string(index=False))
        print(f"\n Лучшая модель: {best_model_name}")
        print(f"Лучший RMSE на тесте: {final_rmse:.3f}")
        print(f"\n Результаты сохранены в папке 'models/'")
        print(f"MLflow run сохранен. Для просмотра выполните: mlflow ui")

        return best_model, results, final_rmse


def run_optuna_study(n_trials=10):
    """Запуск полного исследования Optuna"""
    return train_with_optuna(n_trials=n_trials)


if __name__ == "__main__":

    run_optuna_study(n_trials=10)