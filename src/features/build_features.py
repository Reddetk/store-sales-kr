"""
build_features.py — конструирование признакового пространства (Раздел 2.1 ПЗ).

Схема датасета Corporación Favorita (Kaggle):
    train.csv       : date, store_nbr, family, sales, onpromotion (int — кол-во единиц под акцией)
    oil.csv         : date, dcoilwtico (пропуски в выходные дни)
    holidays_events : date, type, locale, locale_name, description, transferred (bool)
    stores.csv      : store_nbr, city, state, type, cluster
    transactions.csv: date, store_nbr, transactions

Порядок применения:
    1. weekly_aggregation()     — агрегация дневных продаж в недельные
    2. add_lag_features()       — лаговые признаки
    3. add_rolling_features()   — скользящие статистики
    4. add_calendar_features()  — циклические кодировки дат
    5. add_promotion_feature()  — суммарный объём промо за неделю
    6. add_oil_feature()        — нормированная цена нефти
    7. add_holiday_feature()    — бинарный индикатор праздника (с учётом Transfer)
    8. add_transactions_feature() — недельная посещаемость магазина
    9. add_store_features()     — тип и кластер магазина (one-hot)

Каждая функция принимает и возвращает pd.DataFrame.
"""
import numpy as np
import pandas as pd
from src.config import TARGET, DATE_COL, STORE_COL, FAMILY_COL, LAG_WEEKS, ROLLING_WINDOWS


# ── 1. Недельная агрегация ─────────────────────────────────────────────────────
def weekly_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует дневные продажи в недельные суммы по ключам (store_nbr, family).

    Гранулярность train.csv: 1 строка = 1 день × 1 магазин × 1 категория.
    После агрегации: 1 строка = 1 неделя × 1 магазин × 1 категория.

    Правила агрегации:
        sales_weekly  = sum(sales)       — суммарный объём продаж за неделю
        onpromotion   = sum(onpromotion) — суммарное кол-во единиц под акцией за неделю
                        (onpromotion в train.csv — int, не флаг)

    Параметры
    ----------
    df : pd.DataFrame
        train.csv с колонками: date, store_nbr, family, sales, onpromotion.

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
            onpromotion_weekly=("onpromotion", "sum"),  # суммарное кол-во единиц под промо
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
    Добавляет лаговые признаки целевой переменной sales_weekly.

    Лаги lag_1 и lag_4 захватывают краткосрочную автокорреляцию;
    lag_52 кодирует сезонный сигнал прошлого года (S = 52 недели).

    Параметры
    ----------
    df        : pd.DataFrame с колонкой sales_weekly.
    lag_weeks : список сдвигов в неделях; по умолчанию из config [1, 2, 4, 12, 52].

    Возвращает
    ----------
    pd.DataFrame с новыми колонками lag_N.
    """
    if lag_weeks is None:
        lag_weeks = LAG_WEEKS
    df = df.copy().sort_values([STORE_COL, FAMILY_COL, DATE_COL])
    for lag in lag_weeks:
        df[f"lag_{lag}"] = (
            df.groupby([STORE_COL, FAMILY_COL])[TARGET].shift(lag)
        )
    return df


# ── 3. Скользящие статистики ──────────────────────────────────────────────────
def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Добавляет скользящее среднее и стандартное отклонение целевой переменной.

    rolling_mean_4  — локальный тренд за 4 недели.
    rolling_std_4   — волатильность спроса за 4 недели.
    rolling_mean_12 — квартальный сезонный уровень.

    Сдвиг на 1 неделю (shift(1)) перед расчётом предотвращает утечку данных.

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
        df[f"rolling_mean_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"] = grp.transform(
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
    Признак year кодирует линейный тренд 2013–2017.

    Параметры
    ----------
    df : pd.DataFrame с колонкой date.

    Возвращает
    ----------
    pd.DataFrame с пятью новыми колонками.
    """
    df = df.copy()
    dt = pd.to_datetime(df[DATE_COL])
    week  = dt.dt.isocalendar().week.astype(int)
    month = dt.dt.month
    df["week_of_year_sin"] = np.sin(2 * np.pi * week  / 52)
    df["week_of_year_cos"] = np.cos(2 * np.pi * week  / 52)
    df["month_sin"]        = np.sin(2 * np.pi * month / 12)
    df["month_cos"]        = np.cos(2 * np.pi * month / 12)
    df["year"]             = dt.dt.year
    return df


# ── 5. Признак промоакции ──────────────────────────────────────────────────────
def add_promotion_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Нормирует onpromotion_weekly — суммарное количество единиц товара
    под акцией за неделю (уже агрегировано в weekly_aggregation как sum).

    onpromotion в train.csv — int (количество единиц, не флаг).
    Вследствие этого недельное значение = сумма за 7 дней, а не max.
    Добавляется log1p-преобразование для снижения правосторонней асимметрии.

    Параметры
    ----------
    df : pd.DataFrame с колонкой onpromotion_weekly.

    Возвращает
    ----------
    pd.DataFrame с колонкой onpromotion_log1p.
    """
    df = df.copy()
    df["onpromotion_log1p"] = np.log1p(df["onpromotion_weekly"].clip(lower=0))
    return df


# ── 6. Признак цены нефти ──────────────────────────────────────────────────────
def add_oil_feature(
    df: pd.DataFrame,
    oil_df: pd.DataFrame,
    oil_min: float | None = None,
    oil_max: float | None = None,
) -> pd.DataFrame:
    """
    Присоединяет недельную цену нефти WTI и выполняет min-max нормировку.

    oil.csv содержит пропуски в выходные дни. Корректный алгоритм:
        1. Установить date как индекс и привести к полному дневному диапазону
           (asfreq('D')) — это явно добавляет NaN для выходных дней.
        2. Заполнить NaN методом ffill(), затем bfill() для граничных дат.
        3. Агрегировать в недельное среднее по W-MON.
        4. Присоединить к df по колонке date.
        5. Нормировать: oil_norm = (oil - oil_min) / (oil_max - oil_min).

    oil_min и oil_max передаются явно при обработке тестовой выборки,
    чтобы исключить утечку данных (нормировка по обучающей выборке).

    Параметры
    ----------
    df      : pd.DataFrame с колонкой date (W-MON).
    oil_df  : pd.DataFrame с колонками date, dcoilwtico.
    oil_min : минимум из обучающей выборки (None → вычисляется по df).
    oil_max : максимум из обучающей выборки (None → вычисляется по df).

    Возвращает
    ----------
    pd.DataFrame с новыми колонками oil_price и oil_price_norm,
    а также словарём {"oil_min": ..., "oil_max": ...} для передачи в тест.
    """
    oil = oil_df.copy()
    oil[DATE_COL] = pd.to_datetime(oil[DATE_COL])
    # Корректное заполнение: явный дневной индекс → ffill → bfill
    oil = (
        oil.set_index(DATE_COL)
        .resample("D")["dcoilwtico"]
        .mean()             # на случай дублей
        .ffill()
        .bfill()
        .reset_index()
    )
    oil_weekly = (
        oil.groupby(pd.Grouper(key=DATE_COL, freq="W-MON"))["dcoilwtico"]
        .mean()
        .reset_index()
        .rename(columns={"dcoilwtico": "oil_price"})
    )
    df = df.copy()
    df = df.merge(oil_weekly, on=DATE_COL, how="left")
    # Параметры нормировки: передаются явно при работе с тестовой выборкой
    if oil_min is None:
        oil_min = df["oil_price"].min()
    if oil_max is None:
        oil_max = df["oil_price"].max()
    df["oil_price_norm"] = (df["oil_price"] - oil_min) / (oil_max - oil_min)
    return df, oil_min, oil_max


# ── 7. Признак праздников ─────────────────────────────────────────────────────
def add_holiday_feature(
    df: pd.DataFrame,
    holidays_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Добавляет бинарный индикатор наличия значимого праздника в данной неделе.

    Логика holidays_events.csv:
        - transferred=False, type in {Holiday, Additional} → реальный праздник
          по указанной дате (учитывается).
        - transferred=True  → праздник перенесён С этой даты на другую;
          исходная дата НЕ является выходным (НЕ учитывается).
        - type='Transfer'   → дата, НА которую перенесён праздник (учитывается).
        - type in {Bridge, Work Day, Event} → вспомогательные типы,
          не влияющие на базовый спрос (не учитываются в is_holiday,
          но могут быть добавлены отдельными признаками при необходимости).

    Признак is_holiday = 1, если в неделе есть хотя бы один день типа
    Holiday, Additional или Transfer (с transferred=False).

    Параметры
    ----------
    df          : pd.DataFrame с колонкой date (W-MON).
    holidays_df : pd.DataFrame с колонками date, type, locale, locale_name,
                  description, transferred.

    Возвращает
    ----------
    pd.DataFrame с новой колонкой is_holiday (int8, 0 или 1).
    """
    h = holidays_df.copy()
    h[DATE_COL] = pd.to_datetime(h[DATE_COL])
    # Реальные праздники: не перенесённые Holiday/Additional + даты Transfer
    real_holidays = h[
        (h["type"].isin(["Holiday", "Additional"]) & (~h["transferred"])) |
        (h["type"] == "Transfer")
    ].copy()
    # Выравниваем на начало недели W-MON
    real_holidays["week_start"] = (
        real_holidays[DATE_COL]
        - pd.to_timedelta(real_holidays[DATE_COL].dt.weekday, unit="D")
    )
    holiday_weeks = set(real_holidays["week_start"].dt.normalize())
    df = df.copy()
    df["is_holiday"] = (
        pd.to_datetime(df[DATE_COL]).dt.normalize().isin(holiday_weeks)
    ).astype("int8")
    return df


# ── 8. Признак транзакций ─────────────────────────────────────────────────────
def add_transactions_feature(
    df: pd.DataFrame,
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Добавляет недельное число транзакций магазина как прокси посещаемости.

    transactions.csv содержит колонки date, store_nbr, transactions.
    Гранулярность: 1 строка = 1 день × 1 магазин (не разбита по family).
    Вследствие этого признак одинаков для всех семейств товаров
    одного магазина в одну неделю.

    Алгоритм:
        1. Агрегировать дневные транзакции в недельные суммы по store_nbr.
        2. Применить log1p для снижения правосторонней асимметрии.
        3. Присоединить к df по ключам date, store_nbr.

    Параметры
    ----------
    df              : pd.DataFrame с колонками date, store_nbr.
    transactions_df : pd.DataFrame с колонками date, store_nbr, transactions.

    Возвращает
    ----------
    pd.DataFrame с новой колонкой transactions_log1p.
    """
    tx = transactions_df.copy()
    tx[DATE_COL] = pd.to_datetime(tx[DATE_COL])
    tx_weekly = (
        tx.groupby([STORE_COL, pd.Grouper(key=DATE_COL, freq="W-MON")])["transactions"]
        .sum()
        .reset_index()
        .rename(columns={"transactions": "transactions_weekly"})
    )
    tx_weekly["transactions_log1p"] = np.log1p(tx_weekly["transactions_weekly"])
    df = df.copy()
    df = df.merge(
        tx_weekly[[STORE_COL, DATE_COL, "transactions_log1p"]],
        on=[STORE_COL, DATE_COL],
        how="left",
    )
    return df


# ── 9. Признаки магазина ──────────────────────────────────────────────────────
def add_store_features(
    df: pd.DataFrame,
    stores_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Присоединяет метаданные магазина и выполняет one-hot encoding типа магазина.

    stores.csv содержит: store_nbr, city, state, type, cluster.
    Тип магазина (A–E) кодируется через pd.get_dummies (drop_first=False).
    Кластер (1–17) используется как числовой признак.
    city и state не включаются: их информация частично покрыта кластером.

    Параметры
    ----------
    df        : pd.DataFrame с колонкой store_nbr.
    stores_df : pd.DataFrame с колонками store_nbr, city, state, type, cluster.

    Возвращает
    ----------
    pd.DataFrame с новыми колонками cluster и store_type_A … store_type_E.
    """
    stores = stores_df[[STORE_COL, "type", "cluster"]].copy()
    stores = stores.rename(columns={"type": "store_type"})
    df = df.merge(stores, on=STORE_COL, how="left")
    dummies = pd.get_dummies(df["store_type"], prefix="store_type", dtype="int8")
    df = pd.concat([df.drop(columns=["store_type"]), dummies], axis=1)
    return df


# ── Pipeline ──────────────────────────────────────────────────────────────────
def build_feature_matrix(
    train_df: pd.DataFrame,
    oil_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    oil_min: float | None = None,
    oil_max: float | None = None,
) -> tuple[pd.DataFrame, float, float]:
    """
    Сквозной конвейер конструирования признаков (вызывается из ноутбука 01).

    Порядок шагов соответствует Подразделу 2.1 пояснительной записки.
    oil_min / oil_max передаются при обработке тестовой выборки для
    предотвращения утечки данных через нормировку oil_price.

    Возвращает
    ----------
    (pd.DataFrame, oil_min, oil_max)
    """
    df = weekly_aggregation(train_df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_promotion_feature(df)
    df, oil_min, oil_max = add_oil_feature(df, oil_df, oil_min, oil_max)
    df = add_holiday_feature(df, holidays_df)
    df = add_transactions_feature(df, transactions_df)
    df = add_store_features(df, stores_df)
    df = df.dropna(subset=[f"lag_{w}" for w in LAG_WEEKS])
    df = df.reset_index(drop=True)
    return df, oil_min, oil_max
