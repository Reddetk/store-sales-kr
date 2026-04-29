"""
lstm_model.py — построение и обучение LSTM (Подраздел 3.2.3 ПЗ).

Архитектура LSTM для задачи прогнозирования продаж:
    - Входной слой: последовательность длиной SEQ_LEN недель × n_features признаков.
    - LSTM-слой 1: 64 ячейки, return_sequences=True (для стекирования).
    - Dropout(0.2): предотвращение переобучения.
    - LSTM-слой 2: 32 ячейки, return_sequences=False.
    - Dropout(0.2).
    - Dense(1): скалярный прогноз (log1p-шкала).

Целевая переменная: log1p(sales_weekly), как для всех шести моделей (Подраздел 2.4.6 ПЗ).
Оптимизатор: Adam(lr=1e-3).
Функция потерь: MSE (эквивалентна минимизации RMSE на log1p-шкале).
Early stopping: patience=10 по val_loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


SEQ_LEN = 12   # длина входной последовательности (12 недель = 3 месяца)


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Преобразует плоскую таблицу признаков в 3D-тензор для LSTM.

    Для каждой группы (store_nbr, family) строятся скользящие окна длиной seq_len.
    Строка i соответствует признакам недель [i, i+1, ..., i+seq_len-1] и
    целевой переменной недели i+seq_len.

    Параметры
    ----------
    df           : pd.DataFrame с признаками, отсортированный по (store, family, date).
    feature_cols : список признаковых столбцов.
    target_col   : имя целевого столбца (f'target_h{h}').
    seq_len      : длина входного окна.

    Возвращает
    ----------
    (X, y): X.shape = (n_samples, seq_len, n_features), y.shape = (n_samples,).
    """
    from src.config import STORE_COL, FAMILY_COL, DATE_COL
    X_list, y_list = [], []
    for _, group in df.sort_values([STORE_COL, FAMILY_COL, DATE_COL]).groupby(
        [STORE_COL, FAMILY_COL], sort=False
    ):
        feats = group[feature_cols].values.astype(np.float32)
        targets = group[target_col].values.astype(np.float32)
        n = len(feats)
        for i in range(n - seq_len):
            X_list.append(feats[i : i + seq_len])
            y_list.append(targets[i + seq_len])
    if not X_list:
        return np.empty((0, seq_len, len(feature_cols))), np.empty(0)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def build_lstm_model(n_features: int, seq_len: int = SEQ_LEN):
    """
    Строит и компилирует LSTM-модель.

    Параметры
    ----------
    n_features : число признаков (размер последнего измерения входного тензора).
    seq_len    : длина входной последовательности.

    Возвращает
    ----------
    Скомпилированная tf.keras.Model.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow не установлен: pip install tensorflow")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ], name="LSTM_SalesForecast")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def fit_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 512,
    patience: int = 10,
):
    """
    Обучает LSTM с early stopping по val_loss.

    Параметры
    ----------
    X_train, y_train : обучающий тензор.
    X_val, y_val     : валидационный тензор (обычно тестовый период).
    epochs           : максимальное число эпох.
    batch_size       : размер мини-батча.
    patience         : терпение early stopping.

    Возвращает
    ----------
    (обученная модель, объект history).
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow не установлен: pip install tensorflow")

    tf.random.set_seed(42)
    n_features = X_train.shape[2]
    seq_len    = X_train.shape[1]
    model = build_lstm_model(n_features, seq_len)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0,
        ),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history
