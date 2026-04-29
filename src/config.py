"""Central configuration constants for the store-sales-kr project."""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT_DIR / "data" / "raw"
DATA_INT   = ROOT_DIR / "data" / "interim"
DATA_PROC  = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models" / "saved"
REPORTS    = ROOT_DIR / "reports"
FIGURES    = REPORTS / "figures"
TABLES     = REPORTS / "tables"

# ── Target & key columns ─────────────────────────────────────────────────────
TARGET          = "sales_weekly"      # результирующий признак (недельная агрегация)
DATE_COL        = "date"
STORE_COL       = "store_nbr"
FAMILY_COL      = "family"
GROUP_KEYS      = [STORE_COL, FAMILY_COL]

# ── Temporal parameters ─────────────────────────────────────────────────────
SEASONAL_PERIOD   = 52                  # S = 52 недели (годовая сезонность)
FORECAST_HORIZONS = [1, 3, 6, 12]      # горизонты прогнозирования (недели)
# TRAIN_CUTOFF — первый понедельник тестового периода.
# Используется как порог разбивки в ячейке 10: df[date] < cutoff → train, >= cutoff → test.
# ДАТА ОБЯЗАТЕЛЬНО должна быть ПОНЕДЕЛЬНИКОМ: weekly_aggregation использует
# freq="W-MON" — метка недели = понедельник, вследствие чего
# недельная метка train должна быть < cutoff.
# 2017-06-01 (четверг) заменён на 2017-06-05 (понедельник) — неделя
# 2017-05-29–2017-06-04 целиком попадает в train, устраняя AssertionError.
TRAIN_CUTOFF      = "2017-05-01"       # граница обучающей / тестовой выборок (W-MON)

# ── Lag windows ──────────────────────────────────────────────────────────────
LAG_WEEKS       = [1, 2, 4, 9, 12, 52]   # лаги целевой переменной
PROMOTION_LAG   = 1
ROLLING_WINDOWS = [4, 12]            # окна скользящих статистик (недели)

# ── Raw data ranges (пункт 2.2.1 ПЗ) — используются для StandardScaler ──────────────
OIL_PRICE_MIN   = 26.2
OIL_PRICE_MAX   = 110.6
SALES_MIN       = 0.0
SALES_MAX       = 89440.0

# ── Random seed ─────────────────────────────────────────────────────────────────
RANDOM_STATE    = 42
