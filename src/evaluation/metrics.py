"""
metrics.py — метрики качества прогнозирования (Подраздел 3.3 ПЗ).

RMSE, MAE вычисляются на исходной шкале (expm1 от log1p-таргета).
MAPE вычисляется тремя способами для корректной интерпретации разреженных данных:
  - mape(eps=1)  : стандартный MAPE, исключает y_true <= 1.
  - smape(eps=1) : симметричный MAPE (sMAPE), использует тот же фильтр eps=1,
                   что обеспечивает сопоставимость с mape() на разреженных рядах.
  - mape_nonzero : MAPE только на строках y_true >= threshold (default=10).
  - rmsle        : Root Mean Squared Log Error — официальная метрика Kaggle Favorita.

Вследствие структурной разреженности датасета Corporación Favorita (~40–55 %
строк с продажами < 10 ед./нед.) классический MAPE систематически завышается.
Следовательно, в ПЗ основными метриками выступают RMSE и RMSLE,
а sMAPE — как дополнительная интерпретируемая мера (п. 3.3 ПЗ).

Примечание о согласованности фильтров нулей
--------------------------------------------
mape()         исключает строки с y_true <= eps (default eps=1.0).
smape()        применяет тот же фильтр eps=1.0 по умолчанию —
               вследствие этого MAPE и sMAPE вычисляются на одинаковом
               подмножестве наблюдений и остаются сопоставимы.
mape_nonzero() использует более строгий порог threshold=10.0 и предназначен
               для оценки качества на высокообъёмных семействах.
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

    Строки с y_true <= eps исключаются из расчёта вследствие деления на ~0.
    eps=1.0 соответствует минимальному уровню продаж, при котором MAPE
    информативен. Для разреженных данных предпочтительнее smape или mape_nonzero.

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


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """
    Симметричный MAPE (Symmetric Mean Absolute Percentage Error), %.

    sMAPE = mean(200 * |y - ŷ| / (|y| + |ŷ| + ε)) в диапазоне [0, 200 %].

    Строки с y_true <= eps исключаются — тот же фильтр, что у mape(eps=1.0).
    Вследствие единого фильтра MAPE и sMAPE вычисляются на одинаковом
    подмножестве наблюдений: их соотношение характеризует направление смещения
    прогноза (ŷ/y > 1 → систематическое завышение).

    Параметры
    ----------
    y_true : истинные значения на исходной шкале (не log1p).
    y_pred : прогнозные значения на исходной шкале (не log1p).
    eps    : минимальный порог y_true (совпадает с mape по умолчанию).
    """
    mask = y_true > eps
    if mask.sum() == 0:
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    denominator = np.abs(yt) + np.abs(yp) + 1e-8
    return float(100.0 * np.mean(2.0 * np.abs(yt - yp) / denominator))


def mape_nonzero(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 10.0,
) -> float:
    """
    MAPE только на строках с y_true >= threshold.

    Позволяет оценить качество прогноза на высокообъёмных семействах
    (BEVERAGES, PRODUCE) отдельно от структурно разреженных (BOOKS, HARDWARE).
    threshold=10 соответствует нижней границе «значимых» продаж по п. 2.2 ПЗ.

    Параметры
    ----------
    y_true    : истинные значения на исходной шкале.
    y_pred    : прогнозные значения на исходной шкале.
    threshold : минимальный порог y_true.
    """
    mask = y_true >= threshold
    if mask.sum() == 0:
        return float("nan")
    return float(100.0 * np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    ))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Log Error (RMSLE) — официальная метрика Kaggle Favorita.

    RMSLE = sqrt( mean( (log1p(ŷ) - log1p(y))² ) )

    Свойства:
    - Штрафует за недооценку сильнее, чем за переоценку.
    - Инвариантен к абсолютному масштабу: сопоставим между агрегированным
      и детальным уровнями прогноза.
    - Определён при y_true >= 0, y_pred >= 0 (отрицательные значения обрезаются).

    На агрегированном недельном ряде суммарных продаж ≈ 1–3 млн ед. RMSLE
    интерпретируется как средняя логарифмическая ошибка в натуральных единицах:
    RMSLE = 0,30 соответствует ~35 % относительной ошибки в исходном масштабе.

    Параметры
    ----------
    y_true : истинные значения на исходной шкале (не log1p), неотрицательные.
    y_pred : прогнозные значения на исходной шкале (не log1p).
    """
    y_true_c = np.clip(y_true, 0.0, None)
    y_pred_c = np.clip(y_pred, 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true_c) - np.log1p(y_pred_c)) ** 2)))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    log_scale: bool = True,
) -> dict[str, float]:
    """
    Вычисляет RMSE, MAE, MAPE, sMAPE, MAPE_nz, RMSLE для пары (y_true, y_pred).

    Параметры
    ----------
    y_true     : истинные значения.
    y_pred     : прогнозные значения.
    log_scale  : если True, применяет expm1 перед расчётом (модели обучены на log1p).

    Возвращает
    ----------
    Словарь {'RMSE': ..., 'MAE': ..., 'MAPE': ..., 'sMAPE': ...,
             'MAPE_nz': ..., 'RMSLE': ...}.

    Примечание о согласованности MAPE / sMAPE
    ------------------------------------------
    Оба показателя вычисляются на строках y_true > 1.0.
    Вследствие единого фильтра соотношение MAPE/sMAPE характеризует
    направление смещения: MAPE/sMAPE ≈ (y + ŷ) / (2y), откуда
    ŷ/y ≈ 2·(MAPE/sMAPE) - 1. Значение > 1 — систематическое завышение.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if log_scale:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(np.clip(y_pred, -1e6, 20))
    y_pred = np.clip(y_pred, 0, None)
    return {
        "RMSE":    rmse(y_true, y_pred),
        "MAE":     mae(y_true, y_pred),
        "MAPE":    mape(y_true, y_pred, eps=1.0),
        "sMAPE":   smape(y_true, y_pred, eps=1.0),
        "MAPE_nz": mape_nonzero(y_true, y_pred, threshold=10.0),
        "RMSLE":   rmsle(y_true, y_pred),
    }


def metrics_table(
    results: dict[str, dict[int, dict[str, float]]],
) -> pd.DataFrame:
    """
    Строит сводную таблицу метрик: строки = модели, столбцы = горизонт × метрика.

    Параметры
    ----------
    results : {model_name: {horizon: {'RMSE': ..., 'MAE': ..., ...}}}.

    Возвращает
    ----------
    pd.DataFrame с колонками вида h=1_RMSE, h=1_MAE, h=1_MAPE,
    h=1_sMAPE, h=1_MAPE_nz, h=1_RMSLE, ...
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
