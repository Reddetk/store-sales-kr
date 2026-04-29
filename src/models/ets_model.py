"""
ets_model.py — построение модели Хольта-Уинтерса (Подраздел 3.2.6 ПЗ).

Спецификация (Подраздел 2.1.2 ПЗ):
    Модель ETS(A,A,A) — аддитивная ошибка, аддитивный тренд, аддитивная сезонность.
    Сезонный период S = 52 недели (установлен в пункте 2.3.5 ПЗ).

Обоснование выбора аддитивной спецификации:
    СТЛ-декомпозиция (пункт 2.2.2) показала, что амплитуда сезонного компонента
    остаётся приблизительно постоянной при росте тренда, следовательно
    мультипликативная сезонность не требуется.

Область применения:
    Как и SARIMA, Holt-Winters обучается на агрегированном недельном ряде
    суммарных продаж. Внешние регрессоры (oil_price, is_holiday) в базовую
    спецификацию ETS не включаются, поскольку метод не поддерживает
    экзогенные переменные в стандартной реализации statsmodels.
    Данное ограничение зафиксировано в пункте 3.1.2 ПЗ как методологическое
    ограничение эконометрического класса моделей.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


SEASONAL_PERIOD = 52


def fit_holtwinters(
    series: pd.Series,
    seasonal_periods: int = SEASONAL_PERIOD,
    trend: str = "add",
    seasonal: str = "add",
    damped_trend: bool = False,
    use_boxcox: bool = False,
) -> object:
    """
    Обучает модель Хольта-Уинтерса (ETS).

    Параметры
    ----------
    series          : pd.Series с freq='W-MON', log1p-преобразованные продажи.
    seasonal_periods: период сезонности (52 для недельных данных с годовой сезонностью).
    trend           : 'add' — аддитивный тренд; 'mul' — мультипликативный.
    seasonal        : 'add' — аддитивная сезонность; 'mul' — мультипликативная.
    damped_trend    : демпфирование тренда (полезно для долгосрочных горизонтов).
    use_boxcox      : Box-Cox трансформация (не применяется при log1p входе).

    Возвращает
    ----------
    Обученный объект HoltWintersResults.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            series,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            damped_trend=damped_trend,
            use_boxcox=use_boxcox,
            initialization_method="estimated",
        )
        result = model.fit(optimized=True, remove_bias=True)
    return result


def holtwinters_forecast(
    result,
    steps: int,
) -> np.ndarray:
    """
    Генерирует прогноз Хольта-Уинтерса на steps шагов вперёд.

    Параметры
    ----------
    result : обученный HoltWintersResults.
    steps  : горизонт прогноза (число недель).

    Возвращает
    ----------
    np.ndarray прогнозных значений (log1p-шкала, длина = steps).
    """
    forecast = result.forecast(steps)
    return np.asarray(forecast)


def holtwinters_params(result) -> dict:
    """
    Извлекает оптимальные параметры сглаживания из обученной модели.

    Параметры соответствуют Таблице 3.9 ПЗ.

    Возвращает
    ----------
    Словарь {'alpha': ..., 'beta': ..., 'gamma': ..., 'sse': ...}.
    """
    return {
        "alpha": round(result.params.get("smoothing_level", float("nan")), 4),
        "beta":  round(result.params.get("smoothing_trend", float("nan")), 4),
        "gamma": round(result.params.get("smoothing_seasonal", float("nan")), 4),
        "sse":   round(result.sse, 2),
        "aic":   round(result.aic, 2),
    }
