# Assignment 2 - Feature Engineering And Preprocessed Dataset

## Objective

This assignment documents how the raw Walmart sales and weather data are turned
into a supervised modeling dataset.

The final prediction target is:

- `units`: number of units sold for one store, one item, and one date

The preprocessed dataset is saved locally at:

```text
data/processed/modeling_dataset.parquet
```

The processed data is ignored by Git because it is generated from the raw Kaggle
files.

## Raw Data Used

The project uses three raw files from the Walmart Recruiting - Sales in Stormy
Weather Kaggle dataset:

- `train.csv`: historical labeled sales data with `date`, `store_nbr`,
  `item_nbr`, and `units`
- `key.csv`: mapping between each store and its weather station
- `weather.csv`: daily weather observations by station

The tables are merged as follows:

1. `train.csv` is merged with `key.csv` on `store_nbr`
2. the result is merged with cleaned `weather.csv` on `station_nbr` and `date`

This creates one modeling row per store, item, and date.

## Weather Cleaning

The weather file contains coded values:

- `M`: missing
- `-`: unavailable
- `T`: trace precipitation or trace snowfall

Cleaning decisions:

- `T` is converted to `0.005` for precipitation and snowfall
- numeric weather columns are converted from strings to numbers
- missing numeric weather values are filled with station-level medians
- if a station median is unavailable, the global median is used

Some weak weather columns are removed:

- `depart`, because it has many missing/coded values
- `sunrise` and `sunset`, because many values are unavailable
- raw `codesum`, because it is converted into explicit event flags

## Feature Engineering

The pipeline creates these feature groups.

### Identifier Features

- `store_nbr`
- `item_nbr`
- `station_nbr`

These allow the model to learn differences between stores, products, and weather
stations.

### Calendar Features

- year
- month
- day of week
- day of month
- week of year
- quarter
- weekend flag
- month-start flag
- month-end flag
- days since the first observation

These features capture calendar patterns and seasonality.

### Sales History Features

For each store-item pair, the project creates:

- `lag_1`: previous day's sales
- `lag_7`: same weekday last week
- `lag_28`: same weekday four weeks ago
- `rolling_mean_7`: historical 7-day average
- `rolling_mean_28`: historical 28-day average
- `rolling_mean_90`: historical 90-day average

These are the most important features because recent sales are usually the best
signal for future demand.

### Weather Features

The cleaned weather features include temperature, precipitation, snowfall,
pressure, wind speed, and wind direction.

The project also creates event flags from weather codes:

- rain
- snow
- fog
- thunder
- freezing conditions
- count of weather events

Same-day historical weather is used as a proxy for forecast weather. In a real
deployment, the model would use weather forecasts available before replenishment
decisions.

## Leakage Prevention

The main leakage risk is using future sales to predict current sales.

The project avoids this by shifting all lag and rolling sales features before
they are used. For example, the row for a given date never uses `units` from that
same date.

The train/test split is chronological:

- Train: `2012-01-01` to `2014-06-30`
- Test: `2014-07-01` to `2014-10-31`

This matches a real forecasting setup: train on the past, evaluate on the
future.

## Sparse Target Handling

The target is extremely sparse:

- total rows: `4,617,600`
- zero-sales share: about `97.4%`

This matters because a model can look good by predicting values close to zero.
To reduce this issue, training keeps all positive-sales rows and samples a
controlled number of zero-sales rows. Final evaluation is still performed on the
full chronological test set.

The project also reports positive-sales MAE to measure error only when a sale
actually happened.

## Output Files

The preprocessing script is:

```text
scripts/prepare_data.py
```

It creates:

- `data/processed/modeling_dataset.parquet`
- `results/data_overview.csv`
- `results/weather_missingness.csv`
- `results/cleaned_weather_summary.csv`
- `results/monthly_sales.csv`
- `results/monthly_seasonality.csv`
- `results/top_items.csv`
- `results/top_stores.csv`

These files support the modeling scripts, EDA plots, and Streamlit dashboard.
