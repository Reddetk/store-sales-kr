"""
metrics.py — метрики качества прогнозирования (Подраздел 3.3 ПЗ).

RMSE, MAE, MAPE вычисляются на исходной шкале (expm1 от log1p-таргета).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Среднеквадратичная ошибка (Root Mean Squared Error)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Средняя абсолютная ошибка (Mean Absolute Error)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """
    Средняя абсолютная процентная ошибка (Mean Absolute Percentage Error), %.

    Нулевые и близкие к нулю значения y_true добавляют eps для стабильности.
    eps=1.0 соответствует порогу: строки, где y_true < eps, исключаются
    из расчёта MAPE, поскольку деление на ~0 даёт неинформативный выброс.

    Параметры
    ----------
    y_true : истинные значения на исходной шкале (не log1p).
    y_pred : прогнозные значения на исходной шкале (не log1p).
    eps    : минимальный порог y_true для включения строки в расчёт.
    """
    mask = y_true > eps
    if mask.sum() == 0:
        return float("nan")
    return float(100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    log_scale: bool = True,
) -> dict[str, float]:
    """
    Вычисляет RMSE, MAE, MAPE для пары (y_true, y_pred).

    Параметры
    ----------
    y_true     : истинные значения.
    y_pred     : прогнозные значения.
    log_scale  : если True, применяет expm1 перед расчётом (модели обучены на log1p).

    Возвращает
    ----------
    Словарь {'RMSE': ..., 'MAE': ..., 'MAPE': ...}.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if log_scale:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(np.clip(y_pred, -1e6, 20))  # clip to prevent overflow
    y_pred = np.clip(y_pred, 0, None)  # продажи неотрицательны
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE":  mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def metrics_table(
    results: dict[str, dict[int, dict[str, float]]],
) -> pd.DataFrame:
    """
    Строит сводную таблицу метрик: строки = модели, столбцы = горизонт × метрика.

    Параметры
    ----------
    results : {model_name: {horizon: {'RMSE': ..., 'MAE': ..., 'MAPE': ...}}}.

    Возвращает
    ----------
    pd.DataFrame с MultiIndex columns (горизонт, метрика).
    """
    rows = []
    for model, horizons in results.items():
        row = {"Модель": model}
        for h, m in horizons.items():
            for metric, val in m.items():
                row[f"h={h}_{metric}"] = round(val, 4)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Модель")
    return df
