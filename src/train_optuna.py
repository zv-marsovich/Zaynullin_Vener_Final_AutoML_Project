import time
import os

import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.sklearn
import joblib

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from catboost import CatBoostRegressor,  Pool
from catboost import cv as catboost_cv
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

from preprocessing import (
    run_preprocessing, get_features_target, get_columns_types,
    create_preprocessor, split_data
)


def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


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

    # Кросс-валидация с RMSE
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=3, scoring='neg_root_mean_squared_error',
                             n_jobs=-1)

    return -scores.mean()  # Минимизируем RMSE


def objective_catboost(trial, X_train, y_train, cat_cols):
    """Целевая функция Optuna для CatBoost"""
    params = {
        'iterations': trial.suggest_int('iterations', 500, 1500, step=100),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'cat_features': cat_cols,
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
        fold_count=5,
        shuffle=True,
        verbose=False
    )

    return cv_data['test-RMSE-mean'].values[-1]


def objective_lgbm(trial, X_train, y_train, preprocessor):
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

    # Преобразуем категориальные признаки для LightGBM
    X_train_lgb = X_train.copy()
    for col in cat_cols:
        X_train_lgb[col] = X_train_lgb[col].astype('category')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LGBMRegressor(**params))
    ])

    scores = cross_val_score(pipeline, X_train_lgb, y_train,
                             cv=3, scoring='neg_root_mean_squared_error',
                             n_jobs=-1)

    return -scores.mean()


def train_with_optuna(n_trials=30):
    """Обучение всех моделей с Optuna"""

    print("=" * 60)
    print("AutoML с Optuna (байесовская оптимизация)")
    print("=" * 60)

    # Загрузка и предобработка данных
    print("\n1. Загрузка и предобработка данных...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, '..', 'data', 'autos.csv')
    df = run_preprocessing(filepath)
    features, target = get_features_target(df)
    global cat_cols, num_cols
    cat_cols, num_cols = get_columns_types(features)
    X_train, X_test, y_train, y_test = split_data(features, target, test_size=0.2, random_state=42)

    # Создание препроцессора
    preprocessor = create_preprocessor(cat_cols, num_cols)

    results = {}
    best_models = {}

    # 1. Базовый benchmark: LinearRegression
    print("\n📊 Обучение LinearRegression (benchmark)...")
    start_time = time.time()
    lr_pipeline = Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])
    lr_scores = cross_val_score(lr_pipeline, X_train, y_train,
                                cv=3, scoring='neg_root_mean_squared_error')
    lr_rmse = -lr_scores.mean()
    lr_time = time.time() - start_time

    results['LinearRegression'] = {
        'rmse': lr_rmse,
        'train_time': lr_time,
        'best_params': None
    }
    best_models['LinearRegression'] = lr_pipeline

    print(f"   RMSE: {lr_rmse:.3f}, Time: {lr_time:.2f}s")

    # 2. XGBoost с Optuna
    print("\n🚀 Оптимизация XGBoost с Optuna...")
    start_time = time.time()

    study_xgb = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    study_xgb.optimize(
        lambda trial: objective_xgb(trial, X_train, y_train, preprocessor),
        n_trials=n_trials,
        n_jobs=-1,
        show_progress_bar=True
    )

    xgb_time = time.time() - start_time

    # Обучение лучшей модели XGBoost
    best_xgb_params = study_xgb.best_params
    best_xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(**best_xgb_params, random_state=42, objective='reg:squarederror', verbosity=0))
    ])
    best_xgb_pipeline.fit(X_train, y_train)

    # Оценка на кросс-валидации
    xgb_scores = cross_val_score(best_xgb_pipeline, X_train, y_train,
                                 cv=3, scoring='neg_root_mean_squared_error')
    xgb_rmse = -xgb_scores.mean()

    results['XGBoost'] = {
        'rmse': xgb_rmse,
        'train_time': xgb_time,
        'best_params': best_xgb_params,
        'n_trials': n_trials
    }
    best_models['XGBoost'] = best_xgb_pipeline

    print(f"\n   ✅ Лучшие параметры: {best_xgb_params}")
    print(f"   ✅ RMSE: {xgb_rmse:.3f}, Time: {xgb_time:.2f}s")

    # 3. CatBoost с Optuna
    print("\n🚀 Оптимизация CatBoost с Optuna...")
    start_time = time.time()

    study_cat = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=10)
    )

    study_cat.optimize(
        lambda trial: objective_catboost(trial, X_train, y_train, cat_cols),
        n_trials=n_trials,
        n_jobs=-1,
        show_progress_bar=True
    )

    cat_time = time.time() - start_time

    # Обучение лучшей модели CatBoost
    best_cat_params = study_cat.best_params
    best_cat_model = CatBoostRegressor(**best_cat_params, cat_features=cat_cols, random_seed=42, verbose=0)
    best_cat_model.fit(X_train, y_train)

    cat_scores = cross_val_score(best_cat_model, X_train, y_train,
                                 cv=3, scoring='neg_root_mean_squared_error')
    cat_rmse = -cat_scores.mean()

    results['CatBoost'] = {
        'rmse': cat_rmse,
        'train_time': cat_time,
        'best_params': best_cat_params,
        'n_trials': n_trials
    }
    best_models['CatBoost'] = best_cat_model

    print(f"\n   ✅ Лучшие параметры: {best_cat_params}")
    print(f"   ✅ RMSE: {cat_rmse:.3f}, Time: {cat_time:.2f}s")

    # 4. LightGBM с Optuna
    print("\n🚀 Оптимизация LightGBM с Optuna...")
    start_time = time.time()

    study_lgb = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    study_lgb.optimize(
        lambda trial: objective_lgbm(trial, X_train, y_train, preprocessor),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True
    )

    lgb_time = time.time() - start_time

    # Обучение лучшей модели LightGBM
    best_lgb_params = study_lgb.best_params

    # Преобразование категориальных признаков
    X_train_lgb = X_train.copy()
    for col in cat_cols:
        X_train_lgb[col] = X_train_lgb[col].astype('category')

    best_lgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LGBMRegressor(**best_lgb_params, random_state=42, verbose=-1))
    ])
    best_lgb_pipeline.fit(X_train_lgb, y_train)

    lgb_scores = cross_val_score(best_lgb_pipeline, X_train_lgb, y_train,
                                 cv=3, scoring='neg_root_mean_squared_error')
    lgb_rmse = -lgb_scores.mean()

    results['LightGBM'] = {
        'rmse': lgb_rmse,
        'train_time': lgb_time,
        'best_params': best_lgb_params,
        'n_trials': n_trials
    }
    best_models['LightGBM'] = best_lgb_pipeline

    print(f"\n   ✅ Лучшие параметры: {best_lgb_params}")
    print(f"   ✅ RMSE: {lgb_rmse:.3f}, Time: {lgb_time:.2f}s")

    # Выбор лучшей модели
    best_model_name = min(results, key=lambda x: results[x]['rmse'])
    best_model = best_models[best_model_name]

    # Финальное тестирование на тестовой выборке
    if best_model_name in ['CatBoost']:
        X_test_clean = X_test.copy()
        y_pred = best_model.predict(X_test_clean)
    elif best_model_name == 'LightGBM':
        X_test_lgb = X_test.copy()
        for col in cat_cols:
            X_test_lgb[col] = X_test_lgb[col].astype('category')
        y_pred = best_model.predict(X_test_lgb)
    else:
        y_pred = best_model.predict(X_test)

    final_rmse = rmse_scorer(y_test, y_pred)

    # Логирование в MLflow
    mlflow.set_experiment("car_price_prediction_optuna")
    with mlflow.start_run():
        for model_name, metrics in results.items():
            mlflow.log_metric(f"{model_name}_rmse_cv", metrics['rmse'])
            mlflow.log_metric(f"{model_name}_train_time", metrics['train_time'])

        mlflow.log_metric("final_test_rmse", final_rmse)
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_param("optuna_n_trials", n_trials)

        # Сохранение модели
        joblib.dump(best_model, 'models/best_model_optuna.pkl')
        mlflow.log_artifact('models/best_model_optuna.pkl')

    # Вывод результатов
    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)

    results_df = pd.DataFrame([
        {
            'Модель': name,
            'RMSE (CV)': round(metrics['rmse'], 3),
            'Время обучения (сек)': round(metrics['train_time'], 2),
            'Лучшая метрика': '✅' if name == best_model_name else ''
        }
        for name, metrics in results.items()
    ])

    print(results_df.to_string(index=False))
    print(f"\n🏆 Лучшая модель: {best_model_name}")
    print(f"📈 RMSE на тестовой выборке: {final_rmse:.3f}")

    # Визуализация процесса оптимизации
    try:
        fig = optuna.visualization.plot_optimization_history(study_xgb)
        fig.write_image("optuna_history_xgb.png")
        print("\n📊 График оптимизации сохранен: optuna_history_xgb.png")
    except:
        pass

    return best_model, results, final_rmse


def run_optuna_study(n_trials=30):
    """Запуск полного исследования Optuna"""

    return train_with_optuna(n_trials=n_trials)


if __name__ == "__main__":
    # Установка optuna если не установлен

    run_optuna_study(n_trials=30)