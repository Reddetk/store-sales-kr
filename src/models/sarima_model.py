"""
sarima_model.py — построение SARIMA (Подраздел 3.2.5 ПЗ).

Спецификация (выведена в пункте 2.3.4 ПЗ):
    Несезонная часть: ARIMA(1, 1, 0) — AR(1) + первое дифференцирование.
    Сезонная часть: (P=0, D=1, Q=1)_{52} — сезонное дифференцирование + MA(1).
    Итоговая спецификация: SARIMA(1,1,0)(0,1,1)[52].

Обоснование:
    d=1: p-значение ADF = 0,4893 > 0,05, нулевая гипотеза о единичном корне
         не отвергается (Таблица 2.2 ПЗ).
    p=1: ЧАКФ исходного ряда — единственный значимый лаг = 1 (Рисунок 2.9 ПЗ).
    D=1: сезонное дифференцирование для устранения нестационарной сезонности,
         выявленной в нестационарности сезонного компонента STL (Рисунок 2.4 ПЗ).
    Q=1: ACF дифференцированного ряда — пик на лаге 52, PACF на лаге 52 в пределах ДИ
         (Таблица 2.3 ПЗ) → сигнатура SMA(1).

Область применения SARIMA в данной работе:
    Модель обучается на агрегированном недельном ряде суммарных продаж
    (sum по всем магазинам и категориям). Причина: SARIMA требует одного
    временного ряда; обучение 1782 отдельных рядов нецелесообразно
    в рамках академического сравнения. Агрегированный ряд содержит 242 наблюдения,
    что достаточно для оценки SARIMA(1,1,0)(0,1,1)[52] при S+d=53 минимальных.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Финальная спецификация из пункта 2.3.4 ПЗ
SARIMA_ORDER         = (1, 1, 0)
SARIMA_SEASONAL_ORDER = (0, 1, 1, 52)


def fit_sarima(
    series: pd.Series,
    order: tuple = SARIMA_ORDER,
    seasonal_order: tuple = SARIMA_SEASONAL_ORDER,
    disp: bool = False,
) -> object:
    """
    Обучает SARIMA на временном ряде.

    Параметры
    ----------
    series         : pd.Series с freq='W-MON', log1p-преобразованные продажи.
    order          : (p, d, q) несезонная часть.
    seasonal_order : (P, D, Q, S) сезонная часть.
    disp           : вывод лога оптимизации.

    Возвращает
    ----------
    Обученный объект SARIMAXResults.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=disp)
    return result


def sarima_forecast(
    result,
    steps: int,
) -> np.ndarray:
    """
    Генерирует прогноз SARIMA на steps шагов вперёд (исходная log1p-шкала).

    Параметры
    ----------
    result : обученный SARIMAXResults.
    steps  : горизонт прогноза (число недель).

    Возвращает
    ----------
    np.ndarray прогнозных значений (log1p-шкала, длина = steps).
    """
    forecast = result.forecast(steps=steps)
    return np.asarray(forecast)


def sarima_residual_diagnostics(result) -> dict:
    """
    Возвращает диагностику остатков SARIMA (Ljung-Box, нормальность).

    Проверка остатков на автокорреляцию: если p-значение теста Льюнга-Бокса
    превышает 0,05, нулевая гипотеза об отсутствии автокорреляции не отвергается,
    следовательно модель адекватно описывает структуру ряда.

    Возвращает
    ----------
    Словарь с ключами 'ljung_box_p' и 'jarque_bera_p'.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy.stats import jarque_bera

    residuals = result.resid
    lb_result = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    jb_stat, jb_p, *_ = jarque_bera(residuals.dropna())
    return {
        "ljung_box": lb_result,
        "jarque_bera_p": jb_p,
        "aic": result.aic,
        "bic": result.bic,
    }
