"""
scaling.py — стандартизация числовых признаков перед подачей в LSTM и упругую сеть.

Обоснование применения StandardScaler (Подраздел 2.4.6 ПЗ):
    Диапазоны числовых признаков существенно различаются:
        oil_price  : 26,2 – 110,6 (config.OIL_PRICE_MIN/MAX)
        sales_weekly: 0 – 89 440  (config.SALES_MIN/MAX)
    Вследствие этого LSTM и упругая сеть (с L1/L2-регуляризацией)
    требуют сопоставимых масштабов признаков.

    XGBoost и случайный лес инвариантны к масштабу признаков —
    стандартизация для этих моделей НЕ выполняется
    (см. ноутбук 03a_ml_xgboost_rf_elasticnet.ipynb).

    Важное правило: scaler.fit вызывается ТОЛЬКО на обучающей выборке.
    Тестовая выборка трансформируется через transform() без повторного fit.
вследствие чего утечка данных через тест полностью исключается.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def apply_standard_scaler(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    num_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Применяет StandardScaler к числовым признакам обучающей и тестовой выборок.

    Порядок операций:
        1. scaler.fit(df_train[num_cols])         — параметры вычисляются только по train.
        2. df_train[num_cols] = scaler.transform() — трансформация обучающей.
        3. df_test[num_cols]  = scaler.transform() — трансформация тестовой (fit НЕ повторяется).

    Назначение: применяется только для LSTM (03b_ml_lstm.ipynb) и
    упругой сети Elastic Net (03a_ml_xgboost_rf_elasticnet.ipynb).
    XGBoost и случайный лес — без нормализации.

    Типовые num_cols для LSTM/ElasticNet:
        ["sales_weekly", "oil_price", "lag_1", "lag_2", "lag_4", "lag_9",
         "lag_12", "lag_52", "onpromotion_lag1",
         "rolling_mean_4", "rolling_std_4", "rolling_mean_12"]

    Параметры
    ----------
    df_train  : pd.DataFrame обучающей выборки.
    df_test   : pd.DataFrame тестовой выборки.
    num_cols  : список числовых признаков для стандартизации.

    Возвращает
    ----------
    (df_train_scaled, df_test_scaled, fitted_scaler)
    """
    df_train = df_train.copy()
    df_test  = df_test.copy()

    scaler = StandardScaler()
    df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
    df_test[num_cols]  = scaler.transform(df_test[num_cols])

    return df_train, df_test, scaler
