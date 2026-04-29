"""
backtesting.py — пошаговая кросс-валидация на скользящем окне (walk-forward).

Схема (Подраздел 3.1.1 ПЗ):
    - Фиксированное обучающее окно (expanding window): каждый фолд добавляет
      h недель к обучению и оценивает h шагов вперёд.
    - Число фолдов: n_splits (по умолчанию 4).
    - Горизонты прогноза h ∈ {1, 3, 6, 12} недель.

Архитектура прогнозирования (direct multi-step):
    Для каждого горизонта h отдельная модель обучается на матрице признаков,
    где целевая переменная сдвинута на h шагов вперёд:
        target_h[t] = y(t + h)
    Признаки в строке t описывают состояние на момент t (lag_1 = y(t-1), ...).
    Следовательно, модель получает на вход признаки момента t и предсказывает
    значение через h недель вперёд без рекурсивного применения промежуточных прогнозов.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable

from src.config import TARGET, DATE_COL, STORE_COL, FAMILY_COL, FORECAST_HORIZONS


def make_horizon_target(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Конструирует целевую переменную для горизонта h (direct forecasting).

    target_h[t] = log1p(sales_weekly[t + h]), сдвиг по группам (store, family).
    Строки, в которых target_h равен NaN (конец каждой группы), удаляются.

    Параметры
    ----------
    df      : pd.DataFrame с колонками sales_weekly, store_nbr, family, date.
    horizon : горизонт прогноза в неделях.

    Возвращает
    ----------
    pd.DataFrame с колонкой f'target_h{horizon}' (log1p-шкала).
    """
    df = df.copy().sort_values([STORE_COL, FAMILY_COL, DATE_COL])
    col = f"target_h{horizon}"
    # log1p-преобразование целевой переменной: снижает гетероскедастичность
    df[col] = (
        df.groupby([STORE_COL, FAMILY_COL])[TARGET]
        .transform(lambda x: np.log1p(x.shift(-horizon)))
    )
    return df.dropna(subset=[col]).reset_index(drop=True)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Возвращает список признаковых столбцов (исключает служебные и целевые).
    """
    exclude = {
        DATE_COL, STORE_COL, FAMILY_COL, TARGET,
        "onpromotion_weekly", "oil_price",
        "transactions_weekly",
    }
    # Исключаем все target_h* столбцы
    target_cols = {c for c in df.columns if c.startswith("target_h")}
    exclude |= target_cols
    return [c for c in df.columns if c not in exclude]


def train_test_split_by_date(
    df: pd.DataFrame,
    cutoff: str,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разбивает датафрейм по дате TRAIN_CUTOFF для горизонта h.

    Обучающая выборка: строки с date < cutoff.
    Тестовая выборка: строки с date >= cutoff (содержат target_h как будущий y).
    """
    cutoff_ts = pd.Timestamp(cutoff)
    col = f"target_h{horizon}"
    df_h = make_horizon_target(df, horizon)
    train = df_h[df_h[DATE_COL] < cutoff_ts].copy()
    test  = df_h[df_h[DATE_COL] >= cutoff_ts].copy()
    # Убеждаемся, что в тесте нет NaN целевой переменной
    test = test.dropna(subset=[col])
    return train, test
