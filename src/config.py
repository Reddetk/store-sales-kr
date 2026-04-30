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
TRAIN_CUTOFF      = "2017-05-01"       # граница обучающей / тестовой выборок (W-MON)

# ── Lag windows ───────────────────────────────────────────
LAG_WEEKS       = [1, 2, 4, 9, 12, 40, 52]   # lag_40: сезонный якорь для h=12
PROMOTION_LAG   = 1
ROLLING_WINDOWS = [4, 12]

# ── Raw data ranges (пункт 2.2.1 ПЗ) — используются для StandardScaler ──────────────
SALES_MIN       = 0.0
SALES_MAX       = 89440.0

# ── Random seed ─────────────────────────────────────────────────────────────────
RANDOM_STATE    = 42
