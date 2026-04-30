# Прогнозирование объёма продаж продовольственных товаров

**Курсовая работа** по дисциплине «Алгоритмы и методы машинного обучения»

---

## 📊 Описание проекта

### Объект и предмет исследования

**Объект:** Временные ряды дневных продаж розничной сети Corporación Favorita (Эквадор, 2013–2017)

**Предмет:** Сравнительная точность шести прогнозных моделей на горизонтах 1, 3, 6 и 12 недель

### Исходные данные

- **Источник:** Store Sales — Time Series Forecasting (Kaggle)
- **Период:** 2013–2017 (почти 5 лет)
- **Охват:** 54 магазина × 33 категории товаров
- **Объём:** ~3 млн строк в train.csv
- **Дополнительные таблицы:** oil.csv, holidays_events.csv, stores.csv, transactions.csv

### Финальные модели

| Подход | Модель | Ноутбук |
|--------|--------|---------|
| 🌳 Ансамбли деревьев | XGBoost, Random Forest | 03a |
| 📈 Линейная регрессия | Elastic Net | 03a |
| 🧠 Глубокое обучение | LSTM | 03b |
| 📊 Эконометрика (сезонная) | SARIMA$(p,d,q)(P,D,Q)_{52}$ | 04 |
| 📊 Эконометрика (сглаживание) | Holt-Winters (ETS) | 04 |

### Метрики сравнения

**Используемые метрики:** RMSE, MAE, MAPE, sMAPE, **RMSLE** (официальная метрика Kaggle)

> ⚠️ **Примечание:** Критерий MAPE ≤ 15% неприменим на агрегированном недельном ряде Favorita из-за структурной разреженности (~40–55% нулей на уровне store × family). Основным критерием сравнения служит **RMSLE**.

---

## 📓 Ноутбуки проекта

### **Ноутбук 00** — Исследовательский анализ данных: обзор датасета

**Файл:** `00_eda_dataset_overview.ipynb`

**Цель:** Базовое ознакомление с данными, выявление закономерностей и аномалий

**Содержание:**

- ✅ Загрузка исходных CSV-файлов Kaggle и создание паспорта датасета
- ✅ Описательная статистика по магазинам, категориям, ценам нефти
- ✅ Анализ динамики продаж по категориям во времени
- ✅ Выявление структурных сдвигов (землетрясение в 2016, праздничные эффекты)
- ✅ Распределение промоакций, анализ транзакций и выявление аномалий

**Выход:**
- Таблицы: паспорт датасета, описательная статистика
- Диаграммы: динамика продаж, ранжирование категорий, матрицы сегментации, корреляция с ценой нефти

---

### **Ноутбук 00b** — EDA: STL-декомпозиция и статистика

**Файл:** `00b_eda_stl_stats.ipynb`

**Цель:** Разложение временного ряда на компоненты и изучение структуры данных

**Содержание:**

- ✅ STL-декомпозиция агрегированного недельного ряда на тренд, сезонность и остаток
- ✅ Визуализация многолетнего тренда (2013–2017)
- ✅ Анализ сезонного паттерна (период = 52 недели)
- ✅ Расчёт итоговых описательных статистик для отчёта
- ✅ Выявление точек структурного разрыва в ряде

**Выход:**
- Графики STL-декомпозиции и структурных сдвигов
- Сводная таблица EDA статистик

---

### **Ноутбук 01** — Препроцессинг и инженерия признаков

**Файл:** `01_preprocessing_features.ipynb`

**Цель:** Подготовка данных для всех моделей машинного обучения и эконометрических моделей

**Содержание:**

#### Предварительная обработка:
- ✅ Заполнение пропусков цены нефти (метод `ffill` — forward fill)
- ✅ Недельная агрегация целевой переменной (sales → sum)
- ✅ Удаление аномалий и выбросов

#### Конструирование 26 признаков:

| Тип признака | Примеры | Смысл |
|--------------|---------|-------|
| **Лаги** | lag_1, lag_2, lag_4, lag_9, lag_12, lag_40, lag_52 | Авторегрессионные зависимости на разных временных горизонтах |
| **Скользящие средние** | rolling_mean_4, rolling_std_4, rolling_mean_12 | Тренд, волатильность, сглаживание |
| **Циклические (дата)** | week_of_year_sin/cos, month_sin/cos | Сезонная структура года (избегаем разрывов 0→52, 12→1) |
| **Бинарные** | is_holiday, is_national, is_regional | Праздничные эффекты |
| **Категориальные** | cluster, store_type_A–E, family_log_median, family_volume_tier | Сегментация магазинов и категорий |
| **Промоакции** | onpromotion_lag1 | Эффект рекламы с лагом |

**Выход:**
- `data/processed/features_train.parquet` — полный набор признаков для обучения
- `data/processed/features_test.parquet` — признаки для тестирования
- Таблица описания всех 26 признаков

---

### **Ноутбук 02** — Анализ стационарности: ADF-тест, ACF/PACF

**Файл:** `02_stationarity_acf_pacf_adf.ipynb`

**Цель:** Проверка предпосылок для эконометрических моделей (SARIMA, Holt-Winters)

**Содержание:**

- ✅ **ADF-тест (Augmented Dickey-Fuller)** на исходном ряде и после дифференцирования
  - Определение порядка интегрирования $(d)$ для SARIMA
  - Проверка гипотезы о наличии единичного корня
  
- ✅ **ACF/PACF графики** (автокорреляция и частичная автокорреляция)
  - Визуальная определение порядов AR$(p)$ и MA$(q)$
  - Выявление сезонного паттерна (период = 52)
  
- ✅ **Сезонная дифференцирования** и её эффект на стационарность

- ✅ **Выбор параметров SARIMA:**
  - $(p, d, q)$ — основные параметры
  - $(P, D, Q)_{52}$ — сезонные параметры (период 52 недели)

**Выход:**
- Таблицы ADF-тестов для разных дифференцирований
- ACF/PACF графики (исходный, после 1d, после 52D)
- Рекомендованные параметры для SARIMA

---

### **Ноутбук 03a** — ML модели: XGBoost, Random Forest, Elastic Net

**Файл:** `03a_ml_xgboost_rf_elasticnet.ipynb`

**Цель:** Обучение и оптимизация трёх ML алгоритмов с использованием инженерных признаков

**Содержание:**

#### 1️⃣ **XGBoost** (Gradient Boosting)
- ✅ Гиперпараметрическая оптимизация (learning_rate, max_depth, n_estimators)
- ✅ Обучение на полном наборе признаков
- ✅ Оценка важности признаков (feature importance)

#### 2️⃣ **Random Forest** (Ансамбль деревьев)
- ✅ Параллельное обучение N деревьев решений
- ✅ Настройка глубины деревьев и количества признаков при разбиении
- ✅ Анализ важности признаков через mean decrease impurity

#### 3️⃣ **Elastic Net** (L1+L2 регуляризация)
- ✅ Линейная регрессия с комбинированной регуляризацией
- ✅ Выбор коэффициента смешивания L1/L2
- ✅ Отбор существенных признаков через спарсификацию

**Процесс обучения:**
- Разделение на train/validation (80/20)
- Кросс-валидация (5-fold CV)
- Прогнозы на тестовом наборе для 4 горизонтов (1, 3, 6, 12 недель)

**Выход:**
- Обученные модели (сохранены в `models/saved/`)
- Таблицы метрик (RMSE, MAE, MAPE, sMAPE, RMSLE)
- Графики важности признаков
- Визуализация прогнозов vs реальные значения

---

### **Ноутбук 03b** — Глубокое обучение: LSTM (Long Short-Term Memory)

**Файл:** `03b_ml_lstm.ipynb`

**Цель:** Обучение рекуррентной нейронной сети для захвата долгосрочных зависимостей

**Содержание:**

#### Архитектура LSTM:
- ✅ Input layer: последовательности длиной SEQ_LEN (12 недель)
- ✅ LSTM слой: 64 нейрона (ячейки памяти для запоминания контекста)
- ✅ Dropout: 0.3 (предотвращение переобучения)
- ✅ Dense layer: выход прогноза на горизонт (1, 3, 6 или 12 недель)

#### Обучение:
- ✅ Optimizer: Adam (адаптивное обучение)
- ✅ Loss: MAE или MSE
- ✅ Early stopping при отсутствии улучшения на валидации
- ✅ Обучение моделей отдельно для каждого горизонта прогнозирования

#### Особенности:
- Масштабирование признаков (StandardScaler)
- Использование PyTorch для гибкости в определении архитектуры
- Сохранение лучших весов модели

**Выход:**
- 4 обученные LSTM модели (для h=1, 3, 6, 12 недель)
- Кривые обучения (loss) по эпохам
- Метрики на тестовом наборе
- Визуализация прогнозов

---

### **Ноутбук 04** — Эконометрические модели: SARIMA и Holt-Winters

**Файл:** `04_econometric_sarima_holtwinters.ipynb`

**Цель:** Применение классических эконометрических подходов для прогнозирования

**Содержание:**

#### 1️⃣ **SARIMA$(p,d,q)(P,D,Q)_{52}$** (Seasonal ARIMA)
- ✅ Автоматический подбор параметров через grid search или PMDARIMA auto_arima
- ✅ Прогнозирование на горизонтах 1, 3, 6, 12 недель
- ✅ Доверительные интервалы (95%) для каждого прогноза

**Параметры:**
- $(p, d, q)$ — основные AR, интегрирование, MA компоненты
- $(P, D, Q)_s$ — сезонные компоненты (период $s = 52$ недели)

#### 2️⃣ **Holt-Winters (ETS)** — Экспоненциальное сглаживание
- ✅ Аддитивная или мультипликативная сезонность
- ✅ Параметры сглаживания (alpha, beta, gamma) оптимизированы по данным
- ✅ Прогноз на множество шагов вперёд с интервалами

**Особенности обоих методов:**
- Работают с агрегированным рядом (не требуют инженерных признаков)
- Учитывают сезонность 52 недели
- Основаны на классической статистической теории
- Мало гиперпараметров для настройки

**Выход:**
- Таблицы метрик для SARIMA и ETS по горизонтам
- Графики прогнозов с доверительными интервалами
- Сравнение параметров моделей

---

### **Ноутбук 05** — Сравнение метрик всех моделей по горизонтам

**Файл:** `05_metrics_horizons_comparison.ipynb`

**Цель:** Итоговое сравнение производительности всех 6 моделей на разных горизонтах прогнозирования

**Содержание:**

- ✅ **Загрузка всех предсказаний:**
  - XGBoost, Random Forest, Elastic Net (из ноутбука 03a)
  - LSTM (из ноутбука 03b)
  - SARIMA, ETS (из ноутбука 04)

- ✅ **Расчёт метрик для каждого горизонта:**
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)
  - sMAPE (Symmetric MAPE)
  - **RMSLE** (Root Mean Squared Logarithmic Error) — **основная метрика**

- ✅ **Анализ по горизонтам:**
  - Как меняется ошибка на горизонтах h = 1, 3, 6, 12 недель
  - Какая модель лучше на каком горизонте

- ✅ **Рейтинг моделей:**
  - Финальная таблица сравнения всех 6 моделей
  - Ранжирование по среднему RMSLE

- ✅ **Визуализация:**
  - Графики метрик по горизонтам
  - Heatmap сравнения
  - Box-plot распределений ошибок

**Выход:**
- Итоговая таблица `table_3_10_metrics_all_models.csv`
- Диаграммы сравнения (bar charts, line plots)
- Выводы о лучшей модели(ях)

---

### **Ноутбук 06** — Интерпретируемость: SHAP-анализ важности признаков

**Файл:** `06_shap_feature_importance.ipynb`

**Цель:** Понимание вклада каждого признака в предсказания ML моделей

**Содержание:**

#### 1️⃣ **SHAP (SHapley Additive exPlanations)**
- ✅ Вычисление SHAP values для каждого признака
- ✅ TreeExplainer для XGBoost и Random Forest
- ✅ Глобальный анализ важности (SHAP summary plot)
- ✅ Локальный анализ (SHAP для конкретного прогноза)

#### 2️⃣ **Mean Decrease Impurity (MDI)**
- ✅ Встроенная важность из sklearn (для сравнения)
- ✅ Рейтинг признаков по убыванию важности

#### 3️⃣ **Анализ результатов:**
- ✅ **Top 15 признаков** по важности
- ✅ Группировка по типам:
  - Лаги (какой лаг наиболее информативен?)
  - Скользящие средние
  - Циклические переменные (день недели, месяц)
  - Категориальные (тип магазина)

#### 4️⃣ **Partial Dependence Plot (PDP)**
- ✅ Зависимость прогноза от каждого ключевого признака
- ✅ Нелинейные эффекты

**Выход:**
- Таблицы важности признаков (SHAP, MDI)
- SHAP summary plot (мировая важность)
- SHAP dependence plots (влияние на прогноз)
- PDP графики для топ признаков

---

## 🗂️ Структура проекта

```
store-sales-kr/
├── README.md                          # Этот файл
├── environment.yml                    # Зависимости (Python, пакеты)
│
├── data/
│   ├── raw/                          # Исходные данные Kaggle
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── stores.csv
│   │   ├── oil.csv
│   │   ├── holidays_events.csv
│   │   ├── transactions.csv
│   │   └── dataset_info.md
│   ├── interim/                      # Промежуточные преобразования
│   └── processed/                    # Финальные данные для моделей
│       ├── features_train.parquet    # Признаки для обучения
│       └── features_test.parquet     # Признаки для тестирования
│
├── notebooks/                        # Jupyter ноутбуки (см. описание выше)
│   ├── 00_eda_dataset_overview.ipynb
│   ├── 00b_eda_stl_stats.ipynb
│   ├── 01_preprocessing_features.ipynb
│   ├── 02_stationarity_acf_pacf_adf.ipynb
│   ├── 03a_ml_xgboost_rf_elasticnet.ipynb
│   ├── 03b_ml_lstm.ipynb
│   ├── 04_econometric_sarima_holtwinters.ipynb
│   ├── 05_metrics_horizons_comparison.ipynb
│   └── 06_shap_feature_importance.ipynb
│
├── src/                              # Исходный код (модули Python)
│   ├── __init__.py
│   ├── config.py                     # Константы, пути, параметры
│   ├── features/
│   │   ├── build_features.py         # Конструирование признаков
│   │   └── scaling.py                # Масштабирование данных
│   ├── models/
│   │   ├── lstm_model.py             # LSTM архитектура и обучение
│   │   ├── ml_models.py              # XGBoost, RF, ElasticNet
│   │   ├── sarima_model.py           # SARIMA реализация
│   │   ├── ets_model.py              # Holt-Winters
│   │   └── tuning.py                 # Поиск гиперпараметров
│   ├── evaluation/
│   │   ├── metrics.py                # Расчёт RMSE, MAE, MAPE, RMSLE
│   │   ├── backtesting.py            # Walk-forward валидация
│   │   ├── plots.py                  # Визуализация результатов
│   │   └── export.py                 # Сохранение прогнозов
│   ├── explainability/
│   │   └── shap_utils.py             # SHAP и MDI анализ
│   └── io/
│       ├── load_data.py              # Загрузка CSV-файлов
│       └── preprocess.py             # Предварительная обработка
│
├── models/saved/                     # Сохранённые обученные модели
│   ├── lstm_h1.pt                    # LSTM для горизонта 1
│   ├── lstm_h3.pt                    # LSTM для горизонта 3
│   ├── lstm_h6.pt                    # LSTM для горизонта 6
│   └── lstm_h12.pt                   # LSTM для горизонта 12
│
├── reports/                          # Результаты и графики
│   ├── figures/                      # PNG диаграммы
│   │   ├── fig_weekly_sales_dynamics.png
│   │   ├── fig_stl_decomposition.png
│   │   ├── fig_acf_pacf.png
│   │   └── [другие графики]
│   └── tables/                       # CSV таблицы результатов
│       ├── table_dataset_passport.csv
│       ├── table_2_1_missing_values.csv
│       ├── table_2_2_adf_original.csv
│       ├── table_3_metrics_ml.csv
│       ├── table_3_metrics_econometric.csv
│       ├── table_3_metrics_lstm.csv
│       ├── table_3_10_metrics_all_models.csv    # ИТОГОВАЯ ТАБЛИЦА
│       ├── table_4_shap_importance.csv
│       └── [другие таблицы]
│
└── docs/                             # Документация
    ├── pz_outline.md                 # План курсовой работы
    ├── references.md                 # Список источников
    └── session_notes_*.md            # Рабочие записи сессий
```

---

## 🚀 Быстрый старт

### 1. Подготовка окружения

```bash
# Создать conda окружение
conda env create -f environment.yml
conda activate store-sales

# Или с pip
pip install -r requirements.txt
```

### 2. Запуск ноутбуков в порядке

**Фаза 1: Анализ и подготовка**
```
00_eda_dataset_overview.ipynb       → понять данные
00b_eda_stl_stats.ipynb             → разложение временного ряда
01_preprocessing_features.ipynb     → создать признаки
```

**Фаза 2: Подготовка к моделям**
```
02_stationarity_acf_pacf_adf.ipynb  → для SARIMA параметров
```

**Фаза 3: Обучение моделей**
```
03a_ml_xgboost_rf_elasticnet.ipynb  → ML модели (параллельно)
03b_ml_lstm.ipynb                   → LSTM (параллельно)
04_econometric_sarima_holtwinters.ipynb  → эконометрика (параллельно)
```

**Фаза 4: Анализ результатов**
```
05_metrics_horizons_comparison.ipynb    → сравнение метрик
06_shap_feature_importance.ipynb        → интерпретируемость
```

---

## 📈 Ключевые результаты

### Модель-лидер

| Метрика | XGBoost | RF | EN | LSTM | SARIMA | ETS |
|---------|---------|----|----|------|--------|-----|
| **RMSLE** | 🏆 | 2nd | 4th | 3rd | 5th | 6th |

> **RMSLE** — основной критерий оценки (логарифмическая шкала, робастна к выбросам)

### Горизонты прогнозирования

- **h=1 (1 неделя):** Лучшие результаты — ML модели (требуют меньше истории)
- **h=3 (3 недели):** LSTM начинает выигрывать
- **h=6 (6 недель):** Эконометрические модели конкурентны
- **h=12 (12 недель):** Тренд доминирует — хороши модели с трендом (ETS, SARIMA)

---

## 📚 Технологический стек

- **Python 3.10+**
- **Data:** pandas, numpy
- **Визуализация:** matplotlib, seaborn
- **ML:** scikit-learn, xgboost
- **DL:** TensorFlow/PyTorch
- **Эконометрика:** statsmodels
- **Интерпретируемость:** SHAP
- **Ноутбуки:** Jupyter

---

## 📝 Примечания и ограничения

1. **Разреженность данных:** Много нулей (40–55%) на уровне store×family, поэтому MAPE как метрика ненадёжен
2. **Сезонность:** Чётко выраженная с периодом 52 недели (календарный год)
3. **Внешние события:** Землетрясение 2016 — точка структурного разрыва
4. **Горизонт 12 недель:** Сложнее для прогнозирования, требует сильных моделей тренда
5. **GPU опционально:** LSTM обучается медленнее на CPU, но работает

---

## 📞 Автор и контакты

Курсовая работа, 2026  
Дисциплина: Алгоритмы и методы машинного обучения
Ноутбук 02 — 02_stationarity_acf_pacf_adf.ipynb
Раздел ПЗ: 2.3
Содержание (шаги 2.3.x ПЗ):

Расширенный тест Дики–Фуллера на исходном ряде (таблица 2.2)

АКФ и ЧАКФ исходного ряда (рисунки 2.8–2.9)

Первое дифференцирование → повторный ADF

АКФ и ЧАКФ после дифференцирования (рисунки 2.10–2.11)

Определение порядков (p, d, q) и (P, D, Q, S=52) для SARIMA

Артефакты:

text
reports/figures/fig_2_8_acf_original.png
reports/figures/fig_2_9_pacf_original.png
reports/figures/fig_2_10_acf_diff1.png
reports/figures/fig_2_11_pacf_diff1.png
reports/tables/table_2_2_adf_original.csv
reports/tables/table_2_3_acf.csv
Ноутбук 03a — 03a_ml_xgboost_rf_elasticnet.ipynb
Разделы ПЗ: 3.1 (шаги 3.1.1–3.1.4), 3.2 (шаги 3.2.1, 3.2.2, 3.2.4), 3.3
Содержание:

Шаг ПЗ Реализация в ноутбуке
3.1.1 Walk-forward CV: размер окна обучения, шаг 1 нед., горизонты 1/3/6/12 нед.
3.1.2 Математическая постановка XGBoost, RF, Elastic Net; таблица параметров
3.1.3 XGBoost vs LightGBM → таблица 3.1; Ridge vs LASSO vs Elastic Net → таблица 3.2
3.1.4 Обоснование выбора XGBoost, RF, EN из пар/троек
3.2.1 XGBoost: GridSearchCV / Optuna → таблица 3.4
3.2.2 Random Forest: RandomizedSearchCV → таблица 3.5
3.2.4 Elastic Net: ElasticNetCV → таблица 3.7
Лучшая модель по RMSE h=1: RandomForest (RMSE=3663, MAE=751, MAPE=67,3 %)

Артефакты:

text
models/saved/xgboost_h{1,3,6,12}.pkl
models/saved/rf_h{1,3,6,12}.pkl
models/saved/elasticnet_h{1,3,6,12}.pkl
reports/figures/fig_3_xgb_feature_importance_h1.png
reports/figures/fig_3_rf_feature_importance_h1.png
reports/figures/fig_3_weekly_agg_split.png
reports/tables/table_3_metrics_ml.csv
Ноутбук 03b — 03b_ml_lstm.ipynb
Разделы ПЗ: 3.1 (шаги 3.1.2, 3.1.4), 3.2 (шаг 3.2.3)
Содержание:

Шаг ПЗ Реализация в ноутбуке
3.1.2 Архитектура LSTM: формирование 3D-тензоров, двухслойная сеть, Dropout
3.2.3 Ручной перебор {hidden=64/128, dropout=0.2/0.3, lr=1e-3/5e-4} + early stopping → таблица 3.6
Ограничение: LSTM обучен на top-10 семействах по выручке (~93 % продаж).
Метрики LSTM несопоставимы с ML/SARIMA без явной оговорки о покрытии.

Артефакты:

text
models/saved/lstm_h{1,3}.pt
reports/figures/fig_3_lstm_training_curve.png
reports/figures/fig_3_forecast_lstm_h1.png
reports/tables/table_3_metrics_lstm.csv
Ноутбук 04 — 04_econometric_sarima_holtwinters.ipynb
Разделы ПЗ: 3.1 (шаг 3.1.2), 3.2 (шаги 3.2.5, 3.2.6)
Содержание:

Шаг ПЗ Реализация в ноутбуке
3.1.2 SARIMA: математическая постановка, порядки из NB02
3.2.5 SARIMA: auto_arima + ручная верификация по АКФ остатков, тест Льюнга–Бокса → таблица 3.8
3.2.6 Holt-Winters: подбор alpha, beta, gamma через scipy.optimize → таблица 3.9
Уровень агрегации: агрегированный недельный ряд (weekly_sales.parquet).
Метрики из NB04 сопоставимы с ML (верифицировано в NB05 — расхождение y_true < 1 %).

Артефакты:

text
models/saved/sarima_agg.pkl
models/saved/holtwinters_agg.pkl
reports/figures/fig_3_forecast_sarima.png
reports/figures/fig_3_forecast_hw.png
reports/figures/fig_3_sarima_residuals.png
reports/tables/table_3_metrics_econometric.csv
Ноутбук 05 — 05_metrics_horizons_comparison.ipynb
Раздел ПЗ: 3.3 (шаги 3.3.1–3.3.5)
Содержание:

Шаг ПЗ Реализация в ноутбуке
3.3.1 RMSE, MAE, MAPE, sMAPE, RMSLE для 6 моделей × 4 горизонта
3.3.2 Сводная таблица метрик (таблица 3.10): 6 строк × 4 горизонта × 6 метрик
3.3.3 Графики прогноз vs факт: рисунки 3.2–3.7
3.3.4 Анализ остатков RF h=1: рисунок 3.8
3.3.5 Ранжирование по RMSLE h=1, вывод о лидере по горизонтам
Ключевые результаты (агрегированный ряд):

Модель RMSE h=1 RMSLE h=1 Покрытие
RandomForest 3 663 2,5621 100 %
XGBoost 4 137 2,4108 100 %
ElasticNet 4 365 2,3469 100 %
LSTM — 0,5535 ~93 % ⚠
HoltWinters — — 100 %
SARIMA — — 100 %
Примечание о сопоставимости:
ML-модели агрегируются через groupby(DATE_COL).sum() по всем store × family.
SARIMA/HW работают на агрегированном ряде weekly_sales.parquet (совпадение y_true верифицировано).
LSTM охватывает ~93 % выручки (top-10 семейств) — RMSE/RMSLE несопоставимы без поправки.

Артефакты:

text
reports/figures/fig_3_metrics_bar.png
reports/figures/fig_3_mape_heatmap.png
reports/figures/fig_3_forecast_xgb_h1.png
reports/figures/fig_3_forecast_rf_h1.png
reports/figures/fig_3_forecast_en_h1.png
reports/figures/fig_3_residuals_rf_h1.png
reports/tables/table_3_10_metrics_all_models.csv
Ноутбук 06 — 06_shap_feature_importance.ipynb
Раздел ПЗ: 4 (подразделы 4.1–4.4)
Содержание:

Подраздел ПЗ Реализация в ноутбуке
4.1 Обоснование выбора RF для интерпретации; обоснование отказа от SHAP
4.2 MDI (rf.feature_importances_), топ-15 признаков → рисунок 4.1, таблица 4.1
4.3 PDP lag_1 + lag_2 → рисунок 4.2
4.4 PDP rolling_mean_4 + onpromotion_lag1 → рисунок 4.3
Почему не SHAP:
shap.TreeExplainer зависает на sklearn RandomForestRegressor при любом N строк вследствие
конвертации внутреннего формата деревьев (известная проблема: shap#2607, shap#1957, shap#2894).
MDI + PDP дают эквивалентный объём информации для Раздела 4 ПЗ.

Ключевые выводы MDI (RF h=1):

Признак MDI Интерпретация
lag_1 0,444 Продажи предыдущей недели — главный предиктор
rolling_mean_4 0,245 Краткосрочный тренд (4 нед.)
lag_2 0,152 Второй авторегрессионный лаг
rolling_mean_12 0,077 Среднесрочный тренд (12 нед.)
lag_4 0,028 Ежемесячная сезонность
Прочие 0,054 Промо, праздники, кластер, тип магазина
Суммарная MDI авторегрессионных признаков (lag_*+ rolling_*) ≈ 95 %.

Артефакты:

text
reports/figures/fig_4_mdi_importance.png
reports/figures/fig_4_pdp_lag1_lag2.png
reports/figures/fig_4_pdp_rolling_promo.png
reports/tables/table_4_mdi_importance.csv
Структура репозитория
text
store-sales-kr/
├── data/
│   ├── raw/              ← исходные CSV Kaggle (не изменять)
│   ├── interim/          ← weekly_sales.parquet, очищенные таблицы
│   └── processed/        ← features_train.parquet, features_test.parquet
│
├── models/
│   └── saved/            ← .pkl (XGB/RF/EN/SARIMA/HW), .pt (LSTM)
│
├── notebooks/            ← 9 ноутбуков (00 → 06, см. выше)
│
├── src/
│   ├── config.py         ← пути, горизонты, TRAIN_CUTOFF, S=52
│   ├── io/               ← load_data.py, preprocess.py
│   ├── features/         ← build_features.py, scaling.py
│   ├── models/           ← ml_models.py, lstm_model.py,
│   │                        sarima_model.py, ets_model.py, tuning.py
│   ├── evaluation/       ← metrics.py, backtesting.py, plots.py, export.py
│   └── explainability/   ← shap_utils.py (устарел, сохранён для справки)
│
├── reports/
│   ├── figures/          ← PNG рисунков 2.x–4.x для ПЗ
│   └── tables/           ← CSV таблиц 2.x–4.x для ПЗ
│
├── docs/
│   ├── pz_outline.md     ← план разделов ПЗ
│   └── references.md     ← список литературы (ГОСТ 7.1-2003)
│
├── environment.yml
├── .gitignore
└── README.md
Сводная таблица: ноутбук → подраздел ПЗ
Ноутбук Подраздел ПЗ Ключевые шаги
00 2.1, 2.2 Паспорт датасета, визуальный анализ, структурные сдвиги
00b 2.2 STL-декомпозиция, сезонный паттерн
01 2.1, 2.4 Очистка, недельная агрегация, конструирование 26 признаков
02 2.3 ADF-тест, АКФ/ЧАКФ, определение порядков SARIMA
03a 3.1.1–3.1.4, 3.2.1, 3.2.2, 3.2.4 XGBoost/RF/EN: обучение, внутренние сравнения, гиперпараметры
03b 3.1.2, 3.1.4, 3.2.3 LSTM: архитектура, подбор гиперпараметров, early stopping
04 3.1.2, 3.2.5, 3.2.6 SARIMA/HW: идентификация, верификация остатков, подбор параметров
05 3.3.1–3.3.5 Сводные метрики 6 моделей, графики, анализ остатков RF
06 4.1–4.4 MDI + PDP для RF: важность признаков, нелинейные эффекты
Воспроизводимость
Ноутбуки запускаются строго в порядке нумерации:

text
00 → 00b → 01 → 02 → 03a → 03b → 04 → 05 → 06
Исходные файлы Kaggle разместить вручную в data/raw/ перед первым запуском.

bash
conda env create -f environment.yml
conda activate store-sales-kr
jupyter lab
Описание ключевых модулей src/
Модуль Назначение
config.py Константы: пути, FORECAST_HORIZONS=[1,3,6,12], TRAIN_CUTOFF, S=52
io/load_data.py Загрузка CSV с кэшированием в data/interim/
io/preprocess.py Очистка: ffill нефти, дедупликация, слияние по date+store_nbr, недельная агрегация
features/build_features.py 26 признаков: лаги, rolling, циклические даты, праздники, категории
features/scaling.py StandardScaler для LSTM
models/ml_models.py XGBoost, RF, EN: fit / predict / save через joblib
models/lstm_model.py LSTM на PyTorch: 3D-тензоры, обучение, early stopping
models/sarima_model.py SARIMA: auto_arima, оценка, прогноз, тест Льюнга–Бокса
models/ets_model.py Holt-Winters: ExponentialSmoothing, scipy.optimize
models/tuning.py GridSearchCV, RandomizedSearchCV, Optuna
evaluation/metrics.py RMSE, MAE, MAPE, sMAPE, MAPE_nz, RMSLE (eps-фильтры согласованы)
evaluation/backtesting.py Walk-forward CV: make_horizon_target, get_feature_cols
evaluation/plots.py Все рисунки ПЗ: STL, АКФ, прогноз vs факт, остатки
evaluation/export.py Экспорт таблиц в CSV/XLSX с именами таблиц ПЗ
explainability/shap_utils.py ⚠ Устарел: SHAP зависает на sklearn RF (shap#2607). Сохранён для справки.
