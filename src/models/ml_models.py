"""
ml_models.py — обёртки для XGBoost, Random Forest, Elastic Net (Подраздел 3.2 ПЗ).

Каждая функция принимает X_train, y_train, X_test и возвращает y_pred.
Гиперпараметры соответствуют таблицам 3.4–3.7 ПЗ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV
from xgboost import XGBRegressor


# ── XGBoost ────────────────────────────────────────────────────────────────────
def build_xgboost(params: dict | None = None) -> XGBRegressor:
    """
    Строит XGBRegressor с параметрами по умолчанию (подобраны для Favorita).

    Параметры по умолчанию обоснованы:
        n_estimators=500     — достаточно для сходимости на ~400k строках.
        learning_rate=0.05   — консервативный шаг для снижения переобучения.
        max_depth=6          — стандартная глубина для задач ритейла.
        subsample=0.8        — стохастический бустинг снижает дисперсию.
        colsample_bytree=0.8 — случайный подмножество признаков на дерево.
        reg_alpha=0.1        — L1-регуляризация для разреженности.
        reg_lambda=1.0       — L2-регуляризация по умолчанию.
        early_stopping_rounds=50 — остановка при отсутствии улучшения.
    """
    defaults = {
        "n_estimators":      500,
        "learning_rate":     0.05,
        "max_depth":         6,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "random_state":      42,
        "n_jobs":            -1,
        "verbosity":         0,
        "eval_metric":       "rmse",
    }
    if params:
        defaults.update(params)
    return XGBRegressor(**defaults)


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict | None = None,
) -> tuple[XGBRegressor, np.ndarray]:
    """
    Обучает XGBoost с early stopping на валидационной выборке.

    Параметры
    ----------
    X_train, y_train : обучающая выборка.
    X_val, y_val     : валидационная выборка для early stopping (опционально).
    params           : переопределение гиперпараметров.

    Возвращает
    ----------
    (обученная модель, прогноз на X_val или None).
    """
    model = build_xgboost(params)
    if X_val is not None and y_val is not None:
        model.set_params(early_stopping_rounds=50)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        pred = model.predict(X_val)
    else:
        model.set_params(early_stopping_rounds=None)
        model.fit(X_train, y_train)
        pred = None
    return model, pred


# ── Random Forest ──────────────────────────────────────────────────────────────
def build_random_forest(params: dict | None = None) -> RandomForestRegressor:
    """
    Строит RandomForestRegressor с параметрами по умолчанию.

    n_estimators=300   — достаточно для стабилизации ошибки обобщения.
    max_features=0.5   — аналог sqrt для непрерывных признаков.
    min_samples_leaf=5 — ограничение переобучения на разреженных группах.
    """
    defaults = {
        "n_estimators":    300,
        "max_features":    0.5,
        "min_samples_leaf": 5,
        "n_jobs":          -1,
        "random_state":    42,
    }
    if params:
        defaults.update(params)
    return RandomForestRegressor(**defaults)


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> RandomForestRegressor:
    """Обучает Random Forest и возвращает обученную модель."""
    model = build_random_forest(params)
    model.fit(X_train, y_train)
    return model


# ── Elastic Net ────────────────────────────────────────────────────────────────
def fit_elasticnet_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    l1_ratios: list[float] | None = None,
    cv: int = 5,
) -> ElasticNet:
    """
    Подбирает alpha и l1_ratio через ElasticNetCV (перекрёстная проверка).

    ElasticNetCV использует координатный спуск и встроенный путь регуляризации,
    что эффективнее GridSearchCV для задач с непрерывным alpha.

    l1_ratio=1.0 соответствует LASSO; l1_ratio=0 — Ridge.
    Сетка l1_ratios охватывает весь диапазон от Ridge до LASSO, обеспечивая
    выбор оптимального баланса групповой отборки и сжатия.

    Параметры
    ----------
    X_train   : обучающая матрица (нормализованная — обязательно).
    y_train   : log1p-таргет.
    l1_ratios : сетка l1_ratio; по умолчанию [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0].
    cv        : число фолдов кросс-валидации.

    Возвращает
    ----------
    Обученный ElasticNet с оптимальными alpha и l1_ratio.
    """
    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
    en_cv = ElasticNetCV(
        l1_ratio=l1_ratios,
        cv=cv,
        max_iter=5000,
        random_state=42,
        n_jobs=-1,
    )
    en_cv.fit(X_train, y_train)
    print(f"[ElasticNetCV] alpha={en_cv.alpha_:.6f}, l1_ratio={en_cv.l1_ratio_:.3f}")
    # Извлекаем финальную модель с найденными параметрами
    model = ElasticNet(
        alpha=en_cv.alpha_,
        l1_ratio=en_cv.l1_ratio_,
        max_iter=5000,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model
