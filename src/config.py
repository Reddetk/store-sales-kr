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
SEASONAL_PERIOD   = 52                  # S = 52 недели (годовая сезонность)
FORECAST_HORIZONS = [1, 3, 6, 12]      # горизонты прогнозирования (недели)
TRAIN_CUTOFF      = "2017-06-01"       # граница обучающей / тестовой выборок

# ── Lag windows ────────────────────────────────────────────────────────────────
# lag_1, lag_2, lag_4 — значимые лаги 1 и 3 АКФ diff1 (рисунок 2.10 ПЗ).
# lag_9               — единственный значимый лаг ЧАКФ diff1 = 0,32 (рисунок 2.11 ПЗ).
# lag_12              — значимые лаги 12–13 АКФ diff1 (рисунок 2.10 ПЗ).
# lag_52              — АКФ = 0,306 > граница Бартлетта 0,1426 (таблица 2.3 ПЗ).
LAG_WEEKS       = [1, 2, 4, 9, 12, 52]   # лаги целевой переменной

# Лаг признака onpromotion: коэффициент корреляции Пирсона 0,797 (пункт 2.2.1 ПЗ).
PROMOTION_LAG   = 1

ROLLING_WINDOWS = [4, 12]            # окна скользящих статистик (недели)

# ── Raw data ranges (пункт 2.2.1 ПЗ) — используются для обоснования StandardScaler ──
OIL_PRICE_MIN   = 26.2               # минимум dcoilwtico в обучающей выборке
OIL_PRICE_MAX   = 110.6              # максимум dcoilwtico в обучающей выборке
SALES_MIN       = 0.0                # минимум sales_weekly
SALES_MAX       = 89440.0            # максимум sales_weekly

# ── Random seed ────────────────────────────────────────────────────────────────
RANDOM_STATE    = 42
