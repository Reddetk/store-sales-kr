"""
build_features.py — конструирование признакового пространства (Раздел 2.1 ПЗ).

Порядок применения:
    1. weekly_aggregation()  — агрегация дневных продаж в недельные
    2. add_lag_features()    — лаговые признаки
    3. add_rolling_features() — скользящие статистики
    4. add_calendar_features() — циклические кодировки дат
    5. add_promotion_feature() — индикатор промоакции
    6. add_oil_feature()     — нормированная цена нефти
    7. add_holiday_feature() — бинарный индикатор праздника
    8. add_store_features()  — тип и кластер магазина (one-hot)

Каждая функция принимает и возвращает pd.DataFrame.
"""
import numpy as np
import pandas as pd
from src.config import TARGET, DATE_COL, STORE_COL, FAMILY_COL, LAG_WEEKS, ROLLING_WINDOWS


# ── 1. Недельная агрегация ─────────────────────────────────────────────────────
def weekly_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует дневные продажи в недельные суммы по ключам (store_nbr, family).
    Используется функция sum: объём продаж за неделю = сумма дневных объёмов.

    Параметры
    ----------
    df : pd.DataFrame
        Исходный датафрейм с колонками: date, store_nbr, family, sales, onpromotion.

    Возвращает
    ----------
    pd.DataFrame с частотой W-MON (начало недели — понедельник).
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df_weekly = (
        df.groupby([STORE_COL, FAMILY_COL, pd.Grouper(key=DATE_COL, freq="W-MON")])
        .agg(
            sales_weekly=("sales", "sum"),
            onpromotion=("onpromotion", "max"),  # 1, если хоть один день недели — промо
        )
        .reset_index()
    )
    return df_weekly


# ── 2. Лаговые признаки ────────────────────────────────────────────────────────
def add_lag_features(
    df: pd.DataFrame,
    lag_weeks: list[int] | None = None,
) -> pd.DataFrame:
    """
    Добавляет лаговые признаки целевой переменной.

    Лаги lag_1 и lag_4 захватывают краткосрочную автокорреляцию;
    lag_52 кодирует сезонный сигнал прошлого года.

    Параметры
    ----------
    df        : pd.DataFrame с колонкой sales_weekly.
    lag_weeks : список сдвигов в неделях; по умолчанию [1, 2, 4, 12, 52].

    Возвращает
    ----------
    pd.DataFrame с новыми колонками lag_N.
    """
    if lag_weeks is None:
        lag_weeks = LAG_WEEKS
    df = df.copy().sort_values([STORE_COL, FAMILY_COL, DATE_COL])
    for lag in lag_weeks:
        col_name = f"lag_{lag}"
        df[col_name] = (
            df.groupby([STORE_COL, FAMILY_COL])[TARGET]
            .shift(lag)
        )
    return df


# ── 3. Скользящие статистики ──────────────────────────────────────────────────
def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Добавляет скользящее среднее и скользящее стандартное отклонение.

    rolling_mean_4  — среднее за последние 4 недели (локальный тренд).
    rolling_std_4   — волатильность спроса за 4 недели.
    rolling_mean_12 — среднее за квартал (сезонный уровень).

    Параметры
    ----------
    df      : pd.DataFrame с колонкой sales_weekly.
    windows : список размеров окна в неделях; по умолчанию [4, 12].

    Возвращает
    ----------
    pd.DataFrame с новыми колонками rolling_mean_N и rolling_std_N.
    """
    if windows is None:
        windows = ROLLING_WINDOWS
    df = df.copy().sort_values([STORE_COL, FAMILY_COL, DATE_COL])
    for w in windows:
        grp = df.groupby([STORE_COL, FAMILY_COL])[TARGET]
        # min_periods=1 — не отбрасывать строки в начале ряда
        df[f"rolling_mean_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"]  = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std()
        ).fillna(0)
    return df


# ── 4. Циклические кодировки дат ──────────────────────────────────────────────
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Кодирует порядковые признаки дат через sin/cos-преобразования.

    Формулы:
        week_of_year_sin = sin(2π · week / 52)
        week_of_year_cos = cos(2π · week / 52)
        month_sin        = sin(2π · month / 12)
        month_cos        = cos(2π · month / 12)

    Преобразование сохраняет цикличность: расстояние между неделей 1
    и неделей 52 равно расстоянию между любыми соседними неделями.

    Параметры
    ----------
    df : pd.DataFrame с колонкой date.

    Возвращает
    ----------
    pd.DataFrame с четырьмя новыми колонками циклического кодирования
    и признаком year (трендовая компонента).
    """
    df = df.copy()
    dt = pd.to_datetime(df[DATE_COL])
    week  = dt.dt.isocalendar().week.astype(int)
    month = dt.dt.month
    df["week_of_year_sin"] = np.sin(2 * np.pi * week  / 52)
    df["week_of_year_cos"] = np.cos(2 * np.pi * week  / 52)
    df["month_sin"]        = np.sin(2 * np.pi * month / 12)
    df["month_cos"]        = np.cos(2 * np.pi * month / 12)
    df["year"]             = dt.dt.year   # трендовый признак
    return df


# ── 5. Признак промоакции ──────────────────────────────────────────────────────
def add_promotion_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует onpromotion в бинарный int-признак (0 / 1).

    Параметры
    ----------
    df : pd.DataFrame с колонкой onpromotion.

    Возвращает
    ----------
    pd.DataFrame с колонкой onpromotion типа int8.
    """
    df = df.copy()
    df["onpromotion"] = df["onpromotion"].astype(int).astype("int8")
    return df


# ── 6. Признак цены нефти ──────────────────────────────────────────────────────
def add_oil_feature(
    df: pd.DataFrame,
    oil_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Присоединяет недельную цену нефти и выполняет min-max нормировку.

    Алгоритм:
        1. Заполняет пропуски в oil_df методом прямого переноса (ffill)
           и обратного переноса (bfill) — пропуски возникают в выходные дни.
        2. Агрегирует дневные цены в недельные (среднее по неделе).
        3. Присоединяет к df по колонке date.
        4. Нормирует по формуле: oil_norm = (oil - min) / (max - min).

    Параметры
    ----------
    df     : pd.DataFrame с колонкой date.
    oil_df : pd.DataFrame с колонками date, dcoilwtico.

    Возвращает
    ----------
    pd.DataFrame с новыми колонками oil_price и oil_price_norm.
    """
    oil = oil_df.copy()
    oil[DATE_COL] = pd.to_datetime(oil[DATE_COL])
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    # Недельная агрегация (среднее по неделе)
    oil_weekly = (
        oil.groupby(pd.Grouper(key=DATE_COL, freq="W-MON"))["dcoilwtico"]
        .mean()
        .reset_index()
        .rename(columns={"dcoilwtico": "oil_price"})
    )
    df = df.copy()
    df = df.merge(oil_weekly, on=DATE_COL, how="left")
    # Min-max нормировка
    oil_min = df["oil_price"].min()
    oil_max = df["oil_price"].max()
    df["oil_price_norm"] = (df["oil_price"] - oil_min) / (oil_max - oil_min)
    return df


# ── 7. Признак праздников ─────────────────────────────────────────────────────
def add_holiday_feature(
    df: pd.DataFrame,
    holidays_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Добавляет бинарный индикатор наличия праздника в данной неделе.

    Типы событий в holidays_events.csv: Holiday, Event, Additional,
    Bridge, Transfer, Work Day. Признак принимает значение 1, если хотя
    бы один день недели помечен как Holiday или Additional.

    Параметры
    ----------
    df          : pd.DataFrame с колонкой date (начало недели W-MON).
    holidays_df : pd.DataFrame с колонками date, type, transferred.

    Возвращает
    ----------
    pd.DataFrame с новой колонкой is_holiday (int8, 0 или 1).
    """
    h = holidays_df.copy()
    h[DATE_COL] = pd.to_datetime(h[DATE_COL])
    # Оставляем только реальные праздники (не перенесённые)
    h = h[h["transferred"] == False]  # noqa: E712
    h = h[h["type"].isin(["Holiday", "Additional"])]
    # Выравниваем дату на начало недели
    h["week_start"] = h[DATE_COL] - pd.to_timedelta(h[DATE_COL].dt.weekday, unit="D")
    holiday_weeks = set(h["week_start"].dt.strftime("%Y-%m-%d"))
    df = df.copy()
    week_start = pd.to_datetime(df[DATE_COL]).dt.strftime("%Y-%m-%d")
    df["is_holiday"] = week_start.isin(holiday_weeks).astype("int8")
    return df


# ── 8. Признаки магазина ──────────────────────────────────────────────────────
def add_store_features(
    df: pd.DataFrame,
    stores_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Присоединяет метаданные магазина и выполняет one-hot encoding типа магазина.

    Тип магазина (store_type) — категориальный признак A/B/C/D/E;
    кластер (cluster) — порядковый номер 1–17, используется как числовой признак.

    Параметры
    ----------
    df        : pd.DataFrame с колонкой store_nbr.
    stores_df : pd.DataFrame с колонками store_nbr, type, cluster.

    Возвращает
    ----------
    pd.DataFrame с новыми колонками cluster и store_type_A … store_type_E.
    """
    stores = stores_df[[STORE_COL, "type", "cluster"]].copy()
    stores = stores.rename(columns={"type": "store_type"})
    df = df.merge(stores, on=STORE_COL, how="left")
    # One-hot encoding типа магазина (drop_first=False для явной интерпретации)
    dummies = pd.get_dummies(df["store_type"], prefix="store_type", dtype="int8")
    df = pd.concat([df.drop(columns=["store_type"]), dummies], axis=1)
    return df


# ── Pipeline ──────────────────────────────────────────────────────────────────
def build_feature_matrix(
    train_df: pd.DataFrame,
    oil_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    stores_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Сквозной конвейер конструирования признаков (вызывается из ноутбука 01).

    Порядок шагов соответствует Подразделу 2.1 пояснительной записки:
        2.1.1 Результирующий признак: sales_weekly
        2.1.2 Факторные признаки: прямые + конструируемые

    Возвращает
    ----------
    pd.DataFrame, сохраняемый в data/processed/features.parquet.
    """
    df = weekly_aggregation(train_df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_promotion_feature(df)
    df = add_oil_feature(df, oil_df)
    df = add_holiday_feature(df, holidays_df)
    df = add_store_features(df, stores_df)
    # Удаляем строки с NaN в лаговых признаках (первые lag_52 наблюдений)
    df = df.dropna(subset=[f"lag_{w}" for w in LAG_WEEKS])
    df = df.reset_index(drop=True)
    return df
