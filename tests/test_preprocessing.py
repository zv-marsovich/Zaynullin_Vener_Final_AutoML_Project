import os

import pandas as pd
import pytest

from src.preprocessing import (
    load_data, handle_missing_values, remove_duplicates,
    filter_columns, get_features_target, get_columns_types
)


def test_load_data():
    """Тест загрузки данных"""
    test_df = pd.DataFrame({'price': [1000, 2000], 'model': ['BMW', 'Audi']})
    test_df.to_csv('test_data.csv', index=False)

    df = load_data('test_data.csv')
    assert df is not None
    assert 'price' in df.columns
    assert len(df) == 2

    os.remove('test_data.csv')


def test_handle_missing_values():
    """Тест обработки пропусков"""
    df = pd.DataFrame({
        'fuel_type': [None, 'diesel', 'petrol'],
        'vehicle_type': [None, 'bus', 'car'],
        'repaired': [None, 'yes', 'no'],
        'model': ['Audi', 'BMW', 'Mercedes'],
        'gearbox': ['manual', None, 'auto']
    })

    df = handle_missing_values(df)
    assert df['fuel_type'].iloc[0] == 'unknown'
    assert df['repaired'].iloc[0] == 'yes'
    assert df['gearbox'].isna().sum() == 0


def test_filter_columns():
    """Тест фильтрации выбросов"""
    df = pd.DataFrame({
        'price': [0, 5000, 10000, 50000],
        'registration_year': [1900, 2000, 2015, 2020],
        'power': [0, 100, 500, 3000],
        'model': ['Audi', 'BMW', 'Audi', 'Mercedes']
    })

    df = filter_columns(df)
    assert 0 not in df['price'].values
    assert df['registration_year'].max() <= 2017
    assert df['power'].max() <= 2301


def test_get_features_target():
    """Тест разделения на признаки и целевую переменную"""
    df = pd.DataFrame({
        'price': [1000, 2000, 3000],
        'model': ['Audi', 'BMW', 'Mercedes'],
        'year': [2010, 2011, 2012]
    })

    features, target = get_features_target(df)
    assert 'price' not in features.columns
    assert len(target) == 3
    assert features.shape[1] == 2


def test_get_columns_types():
    """Тест определения типов колонок"""
    df = pd.DataFrame({
        'price': [1000, 2000],
        'model': ['Audi', 'BMW'],
        'year': [2010, 2011]
    })

    features, _ = get_features_target(df)
    cat_cols, num_cols = get_columns_types(features)

    assert 'model' in cat_cols
    assert 'year' in num_cols