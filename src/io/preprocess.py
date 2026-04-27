"""
preprocess.py — первичная очистка и слияние исходных CSV-файлов.

Функции:
    load_and_merge()  — загрузка и слияние train.csv + stores.csv + oil.csv
                        + holidays_events.csv + transactions.csv.
    fill_oil_gaps()   — заполнение пропусков в dcoilwtico (выходные дни).
    remove_duplicates() — удаление дублирующихся строк.
    save_interim()    — сохранение результата в data/interim/.
"""
import pandas as pd
from pathlib import Path
from src.config import DATA_RAW, DATA_INT, DATE_COL, STORE_COL, FAMILY_COL


# Файлы, содержащие столбец DATE_COL («date»).
# stores.csv не содержит столбца date — parse_dates для него не передаётся.
_FILES_WITH_DATE = {"train", "test", "oil", "holidays", "transactions"}


def load_raw_files() -> dict[str, pd.DataFrame]:
    """
    Загружает все исходные CSV-файлы из data/raw/.

    Ожидаемые файлы (Kaggle competition: store-sales-time-series-forecasting):
        train.csv, test.csv, stores.csv, oil.csv,
        holidays_events.csv, transactions.csv.

    Возвращает
    ----------
    Словарь {имя_файла_без_расширения: pd.DataFrame}.
    """
    files = {
        "train": "train.csv",
        "test": "test.csv",
        "stores": "stores.csv",
        "oil": "oil.csv",
        "holidays": "holidays_events.csv",
        "transactions": "transactions.csv",
    }
    data = {}
    for key, filename in files.items():
        path = DATA_RAW / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} не найден. "
                f"Загрузите датасет командой:\n"
                f"  kaggle competitions download "
                f"-c store-sales-time-series-forecasting -p data/raw/"
            )
        # parse_dates передаётся только файлам со столбцом DATE_COL
        kwargs = {"parse_dates": [DATE_COL]} if key in _FILES_WITH_DATE else {}
        data[key] = pd.read_csv(path, **kwargs)
    return data


def fill_oil_gaps(oil_df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропуски в колонке dcoilwtico (возникают в выходные дни).

    Алгоритм: установить date как индекс → resample('D') для явного
    создания строк выходных дней → ffill() → bfill() для граничных дат.

    Параметры
    ----------
    oil_df : pd.DataFrame с колонками date, dcoilwtico.

    Возвращает
    ----------
    pd.DataFrame с заполненными пропусками.
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
    return oil


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет дублирующиеся строки из датафрейма.

    Параметры
    ----------
    df : pd.DataFrame.

    Возвращает
    ----------
    pd.DataFrame без дублей.
    """
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"Удалено {n_removed} дублирующихся строк.")
    return df


def save_interim(df: pd.DataFrame, filename: str) -> None:
    """
    Сохраняет датафрейм в data/interim/ в формате parquet.

    Параметры
    ----------
    df       : pd.DataFrame для сохранения.
    filename : имя файла без расширения (добавляется .parquet).
    """
    DATA_INT.mkdir(parents=True, exist_ok=True)
    path = DATA_INT / f"{filename}.parquet"
    df.to_parquet(path, index=False)
    print(f"Сохранено: {path}")
