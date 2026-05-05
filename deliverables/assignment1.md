# Assignment 1 - Walmart Sales Forecasting ML Proof of Concept

## 1. Project Topic

This project is a retail demand forecasting proof of concept. The goal is to
predict how many units of each item will be sold in each Walmart store on each
date, using historical sales and local weather observations.

The business motivation is inventory planning: better forecasts can help a store
avoid stockouts, overstock, and poor replenishment decisions.

## 2. Dataset

- Dataset name: Walmart Recruiting - Sales in Stormy Weather
- Source: Old Kaggle competition dataset
- Local path: `data/raw/walmart-recruiting-sales-in-stormy-weather/`
- Data type: tabular time series
- Target variable: `units`

Raw files used:

- `train.csv`: labeled sales history with `date`, `store_nbr`, `item_nbr`,
  and `units`.
- `key.csv`: mapping between each store and its weather station.
- `weather.csv`: daily station-level weather observations.
- `sampleSubmission.csv`: kept only as a reference for the original Kaggle
  submission format.

The encrypted Kaggle test file was removed because it has no labels and is not
needed for the teacher's evaluation format.

## 3. Initial Dataset Understanding

The labeled training data contains:

- 4,617,600 rows
- 45 stores
- 111 items
- 20 weather stations
- Dates from `2012-01-01` to `2014-10-31`
- 118,696 positive-sales rows
- 97.43% zero-sales rows

The very high zero-sales share is the main modeling difficulty. A model that
predicts near zero can look good on average while still failing on actual demand
days. For that reason, the project reports both global regression metrics and
positive-sales-specific error.

## 4. Data Quality

The weather table contains coded and missing values:

- `M`: missing
- `-`: unavailable
- `T`: trace precipitation or trace snowfall

The data pipeline converts trace precipitation/snowfall to `0.005`, then imputes
numeric weather values using station-level medians. If a station median is not
available, the global median is used as a fallback.

Some weather columns are intentionally dropped:

- `depart`, because it has more than 56% coded/missing values
- `sunrise` and `sunset`, because they have about 47% unavailable values
- raw `codesum`, because it is converted into explicit weather flags instead

## 5. Validation Design

The project uses a chronological split:

- Training period: `2012-01-01` to `2014-06-30`
- Test period: `2014-07-01` to `2014-10-31`

This avoids lookahead leakage. A random split would let the model learn from
future dates, which is not valid for forecasting.

## 6. Feature Engineering

The processed modeling dataset is created by `scripts/prepare_data.py`.

Feature groups:

- identifiers: `store_nbr`, `item_nbr`, `station_nbr`
- date features: year, month, day of week, week of year, quarter, weekend flag,
  month-start flag, month-end flag, days since first observation
- weather measurements: temperature, dew point, wet bulb, heating/cooling degree
  indicators, precipitation, snowfall, pressure, wind speed, wind direction
- weather event flags: rain, snow, fog, thunder, freezing
- sales history features: `lag_1`, `lag_7`, `lag_28`, `rolling_mean_7`,
  `rolling_mean_28`, `rolling_mean_90`

The sales lag and rolling features are shifted by at least one row within each
store-item pair. This means the row for a date never uses that same date's target
value.

## 7. Models

The project compares three registered models:

1. Lag Blend Baseline
   - A transparent benchmark based on previous sales lags and rolling means.

2. Weighted Poisson Regression
   - Count model for nonnegative unit sales.
   - One-hot encoded categorical features.
   - Scaled numeric features.
   - Sampling weights correct the zero/positive prevalence after zero-row
     downsampling.

3. Histogram Gradient Boosting
   - Non-linear tabular model.
   - Uses deterministic categorical dtypes for store, item, station, and calendar
     categories.
   - Trained on `log1p(units)`.

Because the target is very sparse, ML models are trained on all positive-sales
rows plus a controlled sample of zero-sales rows. Evaluation is still performed
on the full chronological test set.

## 8. PCA Experiment

The project includes a separate PCA A/B test in `scripts/run_pca_ab_test.py`.

PCA is tested only on numeric engineered features, not on the full one-hot
categorical space. The experiment compares:

- numeric ridge regression without PCA
- numeric ridge regression with PCA retaining 95% explained variance

PCA is kept as an experiment rather than a default model because dimensionality
reduction can remove useful sparse demand signals.

## 9. Metrics

The project reports:

- MAE
- RMSE
- RMSLE
- R2
- mean actual units
- mean predicted units
- MAE on positive-sales rows
- MAE on zero-sales rows

RMSLE is especially relevant because demand is skewed and the original Kaggle
problem used a forecasting-style error perspective where relative error matters.

## 10. Expected Project Outputs

After running the scripts, the expected outputs are:

- `data/processed/modeling_dataset.parquet`
- `results/data_overview.csv`
- `results/weather_missingness.csv`
- `results/model_metrics.csv`
- `results/pca_ab_test.csv`
- `results/model_predictions_sample.csv`
- charts in `plots/`
- saved model files in `models/`
- Streamlit presentation through `src/app.py`

The main end-to-end command required by the teacher template remains:

```bash
python scripts/main.py
```

The supporting development commands are:

```bash
python scripts/prepare_data.py
python scripts/train_models.py
python scripts/run_pca_ab_test.py
python scripts/generate_plots.py
```

## 11. Current Results

Chronological test period: `2014-07-01` to `2014-10-31`

| Model | MAE | RMSE | RMSLE | R2 | Positive-sales MAE | Zero-sales MAE |
|---|---:|---:|---:|---:|---:|---:|
| Histogram Gradient Boosting | 0.222 | 2.404 | 0.093 | 0.893 | 9.261 | 0.009 |
| Weighted Poisson Regression | 1.404 | 6.598 | 0.623 | 0.197 | 29.725 | 0.738 |
| Lag Blend Baseline | 1.238 | 2.960 | 0.694 | 0.838 | 11.086 | 1.006 |

The best model is Histogram Gradient Boosting. It substantially improves the
lag baseline on every main metric, including RMSLE and positive-sales MAE.

PCA experiment:

| Experiment | Components | Explained variance | MAE | RMSE | RMSLE | R2 |
|---|---:|---:|---:|---:|---:|---:|
| Numeric ridge with PCA | 12 | 95.9% | 1.979 | 68.163 | 0.392 | -87.483 |
| Numeric ridge without PCA | 0 | 0.0% | 1.993 | 68.928 | 0.392 | -89.482 |

PCA slightly improves the numeric ridge experiment, but both numeric-only ridge
variants perform poorly on high-demand rows. PCA is therefore documented as an
experiment, not selected for the final model registry.

## 12. Current Limitations

- The dataset is old and represents a fixed historical competition context.
- The project does not use external holidays or promotions, which would likely
  improve real retail forecasts.
- Store/item demand is extremely sparse, so positive-sales performance remains
  harder than zero-sales performance.
- Weather is station-level, not exact store-level.
