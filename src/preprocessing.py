import re
from typing import List, Tuple

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Загрузка данных"""
    df = pd.read_csv(filepath)
    df = df.rename(
        columns=lambda x: re.sub(
            r'(?<!^)([A-Z])', r'_\1', x
        ).lower())

    return df


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Изменение типа данных столбцов с датами"""
    df['date_crawled'] = pd.to_datetime(df['date_crawled'])
    df['date_created'] = pd.to_datetime(df['date_created'])
    df['last_seen'] = pd.to_datetime(df['last_seen'])

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Обработка пропусков"""
    df["fuel_type"] = df["fuel_type"].fillna("unknown")
    df["vehicle_type"] = df["vehicle_type"].fillna("unknown")
    df["repaired"] = df["repaired"].fillna("yes")
    df.dropna(subset=['model'], inplace=True)
    df.dropna(subset=['gearbox'], inplace=True)

    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Удаление дубликатов"""
    df['fuel_type'] = df['fuel_type'].replace("gasoline", "petrol")
    df = df.drop_duplicates()

    return df

def remove_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Удаление не информативных столбцов , мультиколлинеарных и со слабой корелляцией с целевым признаком"""
    df = df.drop(
        columns=[
            "date_crawled",
            "registration_month",
            "date_created",
            "number_of_pictures",
            "postal_code",
            "last_seen",
            "model"
        ]
    )

    return df


def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Фильтрация столбцов, удаление выбросов и аномальных значений"""
    df = df[df['price'] != 0]
    df = df[(df['registration_year'] >= 1920) & (df['registration_year'] < 2017)]
    df = df[df['power'] < 2301]
    df['power'] = df['power'].replace(0, np.nan)
    df['power'] = df['power'].fillna(df.groupby('model')['power'].transform('median'))

    return df


def get_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Разделение на признаки и целевую переменную"""
    features =  df.drop("price", axis=1)
    target = df["price"]

    return features, target


def get_columns_types(features: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Получение списков категориальных и числовых столбцов"""
    categorical_features = features.select_dtypes(include=['object']).columns.to_list()
    numerical_features = features.select_dtypes(include=['number']).columns.to_list()

    return categorical_features, numerical_features


def create_preprocessor(categorical_features: List, numerical_features: List) -> ColumnTransformer:
    """Создание препроцессора для pipeline"""
    encoder = OneHotEncoder(handle_unknown='ignore', drop='first')
    scaler = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numerical_features),
            ("cat", encoder, categorical_features)
        ], remainder="passthrough"
    )

    return preprocessor

def split_data(
        features: pd.DataFrame,
        target: pd.Series,
        test_size: float=0.2,
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Разделение данных на обучающую и тестовую выборки."""
    return train_test_split(features, target, test_size=test_size, random_state=random_state)

def run_preprocessing(filepath: str) -> pd.DataFrame:
    df = load_data(filepath)
    df = convert_dates(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = filter_columns(df)
    df = remove_columns(df)
    df = df.dropna()
    df = df.reset_index(drop=True)

    return df