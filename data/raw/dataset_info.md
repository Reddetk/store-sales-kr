# Dataset: Store Sales — Time Series Forecasting

Источник: <https://www.kaggle.com/competitions/store-sales-time-series-forecasting>  
Конкурс: Corporación Favorita Grocery Sales Forecasting (Kaggle)  
Период: 2013-01-01 — 2017-08-31  
Магазинов: 54 | Категорий товаров: 33

---

## Файлы датасета

| Файл                  | Строк     | Столбцы                                                        |
|-----------------------|-----------|----------------------------------------------------------------|
| train.csv             | 3 000 888 | date, store_nbr, family, sales, onpromotion                    |
| test.csv              | 28 512    | id, date, store_nbr, family, onpromotion                       |
| stores.csv            | 54        | store_nbr, city, state, type, cluster                          |
| oil.csv               | 1 218     | date, dcoilwtico                                               |
| holidays_events.csv   | 350       | date, type, locale, locale_name, description, transferred      |
| transactions.csv      | 83 488    | date, store_nbr, transactions                                  |
| sample_submission.csv | 3 000 888 | id, sales                                                      |

---

## Описание столбцов

### train.csv

- `date` — дата наблюдения (YYYY-MM-DD)
- `store_nbr` — идентификатор магазина (1–54)
- `family` — категория товаров (33 уникальных значения)
- `sales` — целевая переменная; дневной объём продаж категории в магазине (float)
- `onpromotion` — количество единиц товара категории под акцией в этот день (int)

### test.csv

- `id` — уникальный идентификатор строки
- `date` — период прогноза: 2017-08-16 — 2017-08-31
- `store_nbr`, `family`, `onpromotion` — аналогично train.csv
- Столбец `sales` отсутствует (целевая переменная для предсказания)

### stores.csv

- `store_nbr` — идентификатор магазина
- `city` — город расположения магазина
- `state` — провинция
- `type` — тип магазина (A, B, C, D, E; отражает формат и размер)
- `cluster` — группа магазинов со схожим покупательским поведением (1–17)

### oil.csv

- `date` — дата
- `dcoilwtico` — дневная цена нефти марки WTI (USD); содержит пропуски в выходные дни

### holidays_events.csv

- `date` — дата праздника или события
- `type` — тип: Holiday, Event, Additional, Bridge, Work Day, Transfer
- `locale` — охват: National, Regional, Local
- `locale_name` — название страны, региона или города
- `description` — наименование праздника или события
- `transferred` — булев флаг; True означает официальный перенос праздника на другую дату

### transactions.csv

- `date` — дата
- `store_nbr` — идентификатор магазина
- `transactions` — число транзакций за день (прокси посещаемости магазина)

---

## Примечания

1. Файлы `*.csv` исключены из Git-репозитория (см. `.gitignore`).  
   Для воспроизведения загрузить датасет командой:  
   `kaggle competitions download -c store-sales-time-series-forecasting -p data/raw/`

2. `oil.csv` требует интерполяции пропущенных значений (выходные дни) перед использованием в моделях.

3. В `holidays_events.csv` строки с `transferred = True` обозначают праздник,  
   перенесённый с исходной даты; для корректного признака необходимо учитывать обе даты.

4. Гранулярность train.csv: 1 строка = 1 день × 1 магазин × 1 категория товаров.
