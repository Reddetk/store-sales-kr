"""Central configuration constants for the store-sales-kr project."""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT_DIR / "data" / "raw"
DATA_INT   = ROOT_DIR / "data" / "interim"
DATA_PROC  = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models" / "saved"
REPORTS    = ROOT_DIR / "reports"
FIGURES    = REPORTS / "figures"
TABLES     = REPORTS / "tables"

# ── Target & key columns ───────────────────────────────────────────────────────
TARGET          = "sales_weekly"      # результирующий признак (недельная агрегация)
DATE_COL        = "date"
STORE_COL       = "store_nbr"
FAMILY_COL      = "family"
GROUP_KEYS      = [STORE_COL, FAMILY_COL]

# ── Temporal parameters ────────────────────────────────────────────────────────
SEASONAL_PERIOD = 52                  # S = 52 недели (годовая сезонность)
FORECAST_HORIZONS = [1, 3, 6, 12]    # горизонты прогнозирования (недели)
TRAIN_CUTOFF    = "2017-06-01"        # граница обучающей / тестовой выборок

# ── Lag windows ────────────────────────────────────────────────────────────────
LAG_WEEKS       = [1, 2, 4, 12, 52]  # лаги целевой переменной
ROLLING_WINDOWS = [4, 12]            # окна скользящих статистик (недели)

# ── Random seed ────────────────────────────────────────────────────────────────
RANDOM_STATE    = 42
