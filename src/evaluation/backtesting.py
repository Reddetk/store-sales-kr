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
    target_col: str = TARGET,
) -> pd.DataFrame:
    """
    Конструирует целевую переменную для горизонта h (direct forecasting).

    target_h[t] = log1p(target_col[t + h]), сдвиг по группам (store, family).
    Строки, в которых target_h равен NaN (конец каждой группы), удаляются.

    Параметры
    ----------
    df         : pd.DataFrame с колонками sales_weekly, store_nbr, family, date.
    horizon    : горизонт прогноза в неделях.
    target_col : базовая колонка для сдвига (по умолчанию TARGET из config).
                 Позволяет явно передавать имя колонки без изменения конфига.

    Возвращает
    ----------
    pd.DataFrame с колонкой f'target_h{horizon}' (log1p-шкала).
    """
    df = df.copy().sort_values([STORE_COL, FAMILY_COL, DATE_COL])
    col = f"target_h{horizon}"
    # log1p-преобразование целевой переменной: снижает гетероскедастичность
    df[col] = (
        df.groupby([STORE_COL, FAMILY_COL])[target_col]
        .transform(lambda x: np.log1p(x.shift(-horizon)))
    )
    return df.dropna(subset=[col]).reset_index(drop=True)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Возвращает список признаковых столбцов (25 шт., Таблица 2.4 ПЗ).

    Семейства товаров представлены двумя объёмными признаками:
        family_log_median  — log1p(медиана недельных продаж по семейству)
        family_volume_tier — 1 для BEVERAGES/PRODUCE, 0 для остальных

    Поскольку parquet содержит 33 legacy one-hot столбца family_*,
    они исключаются явно: их информация поглощена family_log_median
    и family_volume_tier. Следовательно, FEATURE_COLS = 25 признаков.
    """
    _VOLUME_FEATURES = {"family_log_median", "family_volume_tier"}
    exclude = {
        DATE_COL, STORE_COL, FAMILY_COL, TARGET,
        "onpromotion_weekly", "oil_price",
        "transactions_weekly",
        "rolling_std_12",   # вспомогательный, не входит в реестр 25
        "year",             # поглощается sin/cos цикличными признаками
    }
    # Исключаем target_h*
    target_cols = {c for c in df.columns if c.startswith("target_h")}
    exclude |= target_cols
    # Исключаем legacy family one-hot (family_AUTOMOTIVE и т.д.)
    # — оставляем только family_log_median и family_volume_tier
    family_onehot = {
        c for c in df.columns
        if c.startswith("family_") and c not in _VOLUME_FEATURES
    }
    exclude |= family_onehot
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
