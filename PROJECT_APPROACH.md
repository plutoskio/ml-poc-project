# Walmart Sales Forecasting ML Proof of Concept

## Objective

Build a supervised regression proof of concept that forecasts daily item-level
unit sales for Walmart stores using historical sales, calendar variables, and
local weather observations.

## Teacher Template Alignment

The project keeps the required template contracts:

- `src/data.py` exposes `load_dataset_split()`.
- `src/metrics.py` exposes `compute_metrics(y_true, y_pred)`.
- `src/config.py` registers saved models in `MODELS`.
- `scripts/main.py` evaluates registered models and launches Streamlit.
- `src/app.py` contains the fixed Streamlit entry point `build_app()`.

## Evaluation Design

The original Kaggle test file does not include labels, so the class project uses
a chronological validation split from the labeled training data:

- Training period: `2012-01-01` to `2014-06-30`
- Test period: `2014-07-01` to `2014-10-31`

This is more realistic than a random split because demand forecasting should not
train on future dates.

## Feature Engineering

The pipeline creates:

- calendar features such as month, day of week, week of year, weekend, and month
  boundary flags;
- store, item, and weather station identifiers;
- cleaned weather measurements;
- weather event flags for rain, snow, fog, thunder, and freezing conditions;
- lag and rolling sales features for each store-item pair.

Lag features are shifted before rolling calculations, so each row only sees
historical sales values.

Same-day historical weather is used as a proxy for same-day weather forecasts.
In a production forecasting setup, future weather inputs would come from a
forecast provider before the sales day occurs, so these weather variables are not
treated as leakage in the way same-day sales would be.

## Modeling

The proof of concept compares:

- a transparent lag-based baseline;
- a weighted Poisson regression model for nonnegative count targets;
- a histogram gradient boosting model on engineered tabular features.

The target distribution is extremely sparse, so ML models train on all positive
sales rows plus a sampled set of zero-sales rows. The Poisson model receives
sampling weights so the fitted objective better reflects the original
zero/positive prevalence. Final evaluation still runs on the full chronological
test split.

## Metrics

The project reports:

- MAE;
- RMSE;
- RMSLE;
- R2;
- MAE on positive-sales rows;
- MAE on zero-sales rows;
- average actual and predicted units.

RMSLE is included because the original Kaggle-style forecasting task penalizes
relative errors and handles the skewed unit distribution better than RMSE alone.

## Results

On the chronological test split from `2014-07-01` to `2014-10-31`, Histogram
Gradient Boosting is the best model:

- MAE: 0.222
- RMSE: 2.404
- RMSLE: 0.093
- R2: 0.893
- positive-sales MAE: 9.261
- zero-sales MAE: 0.009

The weighted Poisson model is a useful count-model comparison but does not beat
the lag baseline on RMSE or positive-sales MAE. The PCA A/B test slightly
improves numeric ridge RMSLE, but both PCA and non-PCA numeric ridge variants
perform poorly on high-demand rows, so PCA is not selected for the final model.
