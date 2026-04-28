"""
build_features.py — конструирование признакового пространства (Раздел 2.4 ПЗ).

Схема датасета Corporación Favorita (Kaggle):
    train.csv       : date, store_nbr, family, sales, onpromotion (int — кол-во единиц под акцией)
    oil.csv         : date, dcoilwtico (пропуски в выходные дни)
    holidays_events : date, type, locale, locale_name, description, transferred (bool)
    stores.csv      : store_nbr, city, state, type, cluster
    transactions.csv: date, store_nbr, transactions

Порядок применения:
    1. weekly_aggregation()       — агрегация дневных продаж в недельные
    2. add_lag_features()         — лаговые признаки (включая lag_9, onpromotion_lag1)
    3. add_rolling_features()     — скользящие статистики (min_periods=2 для std)
    4. add_calendar_features()    — циклические кодировки дат (период 52: АКФ=0,306 > 0,1426)
    5. add_promotion_feature()    — суммарный объём промо за неделю
    6. add_oil_feature()          — нормированная цена нефти
    7. add_holiday_feature()      — is_holiday + is_national + is_regional
    8. add_transactions_feature() — недельная посещаемость магазина
    9. add_store_features()       — тип и кластер магазина (one-hot); применяется ДО разбивки train/test

Кодирование категориальных признаков (add_store_features) выполняется до dropna
и до разбивки train/test (см. ноутбук 01, ячейка 10), чтобы гарантировать
одинаковое признаковое пространство в обоих подмножествах.

Нормализация числовых признаков (StandardScaler) вынесена в отдельный модуль
src/features/scaling.py и применяется только для LSTM и упругой сети.
XGBoost и случайный лес инвариантны к масштабу признаков и не требуют нормализации.

Итоговое признаковое пространство (22 признака, таблица 2.4 ПЗ):
    Лаги (6)       : lag_1, lag_2, lag_4, lag_9, lag_12, lag_52, onpromotion_lag1
    Скользящие (3) : rolling_mean_4, rolling_std_4, rolling_mean_12
    Циклические (4): week_of_year_sin, week_of_year_cos, month_sin, month_cos
    Праздники (3)  : is_holiday, is_national, is_regional
    Магазин (6)    : store_type_A … store_type_E, cluster
    Итого: 22 признака (rolling_std_12 — вспомогательный, не входит в итоговое пространство)
"""
import numpy as np
import pandas as pd
from src.config import (
    TARGET, DATE_COL, STORE_COL, FAMILY_COL,
    LAG_WEEKS, ROLLING_WINDOWS, PROMOTION_LAG,
)


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
            onpromotion_weekly=("onpromotion", "sum"),
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
    Добавляет лаговые признаки целевой переменной sales_weekly и признак onpromotion_lag1.

    Обоснование лагов (рисунки 2.10, 2.11 ПЗ; таблица 2.3 ПЗ):
        lag_1, lag_2  — значимые лаги 1–2 АКФ diff1 (рисунок 2.10 ПЗ).
        lag_4         — значимый лаг 4 АКФ diff1 (рисунок 2.10 ПЗ).
        lag_9         — единственный значимый лаг ЧАКФ diff1 = 0,32 (рисунок 2.11 ПЗ).
        lag_12        — значимые лаги 12–13 АКФ diff1 (рисунок 2.10 ПЗ).
        lag_52        — АКФ = 0,306 > граница Бартлетта 0,1426 (таблица 2.3 ПЗ).

    Признак onpromotion_lag1:
        Сдвиг onpromotion_weekly на 1 неделю вперёд обоснован корреляцией Пирсона
        0,797 между onpromotion и sales_weekly (пункт 2.2.1 ПЗ). Лаг предотвращает
        утечку данных: промо текущей недели недоступно на момент прогноза.

    Параметры
    ----------
    df        : pd.DataFrame с колонками sales_weekly, onpromotion_weekly.
    lag_weeks : список сдвигов в неделях; по умолчанию из config [1, 2, 4, 9, 12, 52].

    Возвращает
    ----------
    pd.DataFrame с новыми колонками lag_N и onpromotion_lag1.
    """
    if lag_weeks is None:
        lag_weeks = LAG_WEEKS
    df = df.copy().sort_values([STORE_COL, FAMILY_COL, DATE_COL])
    for lag in lag_weeks:
        df[f"lag_{lag}"] = (
            df.groupby([STORE_COL, FAMILY_COL])[TARGET].shift(lag)
        )
    # onpromotion_lag1: лаг первого порядка признака промоакции
    # Обоснование: корреляция Пирсона 0,797 (пункт 2.2.1 ПЗ)
    df["onpromotion_lag1"] = (
        df.groupby([STORE_COL, FAMILY_COL])["onpromotion_weekly"].shift(PROMOTION_LAG)
    )
    return df


# ── 3. Скользящие статистики ──────────────────────────────────────────────────
def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Добавляет скользящее среднее и стандартное отклонение целевой переменной.

    rolling_mean_4  — локальный тренд за 4 недели (входит в итоговое пространство).
    rolling_std_4   — волатильность спроса за 4 недели (входит в итоговое пространство).
    rolling_mean_12 — квартальный сезонный уровень (входит в итоговое пространство).
    rolling_std_12  — вспомогательный признак, НЕ входит в итоговое пространство
                      таблицы 2.4 ПЗ; генерируется для аналитических целей.

    Сдвиг на 1 неделю (shift(1)) перед расчётом предотвращает утечку данных.

    Исправление FIX-2 (синхронизировано с ноутбуком 01, ячейка 9):
        min_periods для rolling std установлен равным 2, поскольку стандартное
        отклонение по одному наблюдению не определено статистически.

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
        # FIX-2: min_periods=2 для std (синхронизировано с NB01, ячейка 9)
        df[f"rolling_std_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).std()
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

    Обоснование периода 52 (таблица 2.3 ПЗ):
        АКФ на лаге 52 = 0,306 превышает границу Бартлетта 0,1426,
        что подтверждает статистически значимую годовую сезонность.
        Числовые значения зафиксированы в reports/tables/table_2_3_acf.csv.

    Циклическое кодирование необходимо для LSTM и упругой сети, поскольку
    эти модели не различают порядковый и непрерывный характер признаков:
    без sin/cos кодирования расстояние между неделей 1 и неделей 52
    воспринимается как максимальное, а не нулевое.
    САРИМА моделирует сезонность через параметр S = 52 явно и не требует
    циклического кодирования календарных признаков.

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

    Диапазоны цены нефти и продаж (пункт 2.2.1 ПЗ):
        dcoilwtico : 26,2 – 110,6 (зафиксировано в config.OIL_PRICE_MIN/MAX)
        sales_weekly: 0 – 89 440  (зафиксировано в config.SALES_MIN/MAX)

    oil.csv содержит пропуски в выходные дни. Алгоритм:
        1. Установить date как индекс и привести к полному дневному диапазону
           (asfreq('D')) — добавляет NaN для выходных дней явно.
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
    а также словарём {\"oil_min\": ..., \"oil_max\": ...} для передачи в тест.
    """
    oil = oil_df.copy()
    oil[DATE_COL] = pd.to_datetime(oil[DATE_COL])
    oil = (
        oil.set_index(DATE_COL)
        .resample("D")["dcoilwtico"]
        .mean()
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
    Добавляет три бинарных индикатора праздников по локали.

    Признаки:
        is_holiday  — объединение всех типов (National + Regional + Local);
                      = 1, если в неделе есть хотя бы один реальный праздник любого масштаба.
        is_national — только locale == 'National'; 174 записи в holidays_events.csv;
                      обоснование: рождественские пики в STL-декомпозиции (пункт 2.3 ПЗ).
        is_regional — только locale == 'Regional'; 24 события;
                      обоснование: локальные скачки продаж в отдельных провинциях (пункт 2.3 ПЗ).

    Логика holidays_events.csv:
        - transferred=False, type in {Holiday, Additional} → реальный праздник (учитывается).
        - transferred=True  → праздник перенесён С этой даты (НЕ учитывается).
        - type='Transfer'   → дата, НА которую перенесён праздник (учитывается).
        - type in {Bridge, Work Day, Event} → не учитываются в базовых признаках.

    Примечание о пересечении is_national и is_regional:
        На недельном уровне агрегации пересечение is_national=1 и is_regional=1
        в одной строке допустимо: национальный и региональный праздники могут
        попадать в разные дни одной и той же недели. Это не ошибка данных.

    Параметры
    ----------
    df          : pd.DataFrame с колонкой date (W-MON).
    holidays_df : pd.DataFrame с колонками date, type, locale, transferred.

    Возвращает
    ----------
    pd.DataFrame с новыми колонками is_holiday, is_national, is_regional (int8).
    """
    h = holidays_df.copy()
    h[DATE_COL] = pd.to_datetime(h[DATE_COL])

    # Реальные праздники: не перенесённые Holiday/Additional + даты Transfer
    real_holidays = h[
        (h["type"].isin(["Holiday", "Additional"]) & (~h["transferred"])) |
        (h["type"] == "Transfer")
    ].copy()

    def _to_week_set(mask: pd.Series) -> set:
        sub = real_holidays[mask].copy()
        sub["week_start"] = (
            sub[DATE_COL] - pd.to_timedelta(sub[DATE_COL].dt.weekday, unit="D")
        )
        return set(sub["week_start"].dt.normalize())

    all_weeks      = _to_week_set(pd.Series([True] * len(real_holidays), index=real_holidays.index))
    national_weeks = _to_week_set(real_holidays["locale"] == "National")
    regional_weeks = _to_week_set(real_holidays["locale"] == "Regional")

    df = df.copy()
    dates_norm = pd.to_datetime(df[DATE_COL]).dt.normalize()

    df["is_holiday"]  = dates_norm.isin(all_weeks).astype("int8")
    df["is_national"] = dates_norm.isin(national_weeks).astype("int8")
    df["is_regional"] = dates_norm.isin(regional_weeks).astype("int8")

    # FIX-3: пересечение is_national и is_regional допустимо на недельном уровне
    # (национальный и региональный праздники в разные дни одной недели).
    # Выводим число таких недель как информационное сообщение, не как ошибку.
    n_overlap = int((df["is_national"].astype(bool) & df["is_regional"].astype(bool)).sum())
    if n_overlap > 0:
        print(
            f"[INFO] add_holiday_feature: {n_overlap} строк с is_national=1 и is_regional=1 "
            "(нац. и рег. праздники в одну неделю — допустимо на W-MON агрегации)"
        )

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
    Результат: 5 бинарных столбцов store_type_A … store_type_E.
    Кластер (1–17) используется как числовой признак.
    city и state не включаются: их информация частично покрыта кластером.

    Обоснование включения типа магазина (пункт 2.2.1 ПЗ):
        Тип магазина объясняет 0,33–0,64 доли вариации продаж по семействам
        товаров (таблица/рисунок пункта 2.2.1 ПЗ).

    ВАЖНО: функция вызывается ДО операции dropna и ДО разбивки train/test
    (см. build_feature_matrix → add_store_features → dropna → ноутбук 01, ячейка 10),
    что обеспечивает одинаковое пространство признаков в обоих подмножествах.

    Параметры
    ----------
    df        : pd.DataFrame с колонкой store_nbr.
    stores_df : pd.DataFrame с колонками store_nbr, city, state, type, cluster.

    Возвращает
    ----------
    pd.DataFrame с новыми колонками cluster и store_type_A … store_type_E (5 столбцов).
    """
    stores = stores_df[[STORE_COL, "type", "cluster"]].copy()
    stores = stores.rename(columns={"type": "store_type"})
    df = df.merge(stores, on=STORE_COL, how="left")
    dummies = pd.get_dummies(df["store_type"], prefix="store_type", dtype="int8")
    # Контроль числа one-hot столбцов: ожидается 5 (типы A–E)
    assert len(dummies.columns) == 5, (
        f"Ожидалось 5 one-hot столбцов store_type, получено {len(dummies.columns)}"
    )
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

    Порядок шагов соответствует Подразделу 2.4 пояснительной записки.
    oil_min / oil_max передаются при обработке тестовой выборки для
    предотвращения утечки данных через нормировку oil_price.

    Итоговое число признаков после dropna: 22
        6 лагов (lag_1, lag_2, lag_4, lag_9, lag_12, lag_52) + onpromotion_lag1 = 7
        3 скользящих (rolling_mean_4, rolling_std_4, rolling_mean_12)
        4 циклических (week_of_year_sin/cos, month_sin/cos)
        3 праздничных (is_holiday, is_national, is_regional)
        5 store_type (A–E) + cluster = 6
        Итого: 7 + 3 + 4 + 3 + 6 = 23 признака
        (rolling_std_12 вспомогательный — итог зависит от финального отбора)

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
    # add_store_features вызывается ДО dropna — см. docstring функции
    df = add_store_features(df, stores_df)
    df = df.dropna(subset=[f"lag_{w}" for w in LAG_WEEKS])
    df = df.reset_index(drop=True)
    return df, oil_min, oil_max
