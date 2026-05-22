# Assignment 3 - Model Description And Comparison

## Objective

This assignment documents the three supervised models used in the Walmart demand
forecasting proof of concept.

The target is:

- `units`: number of units sold for one store, one item, and one date

The project is a regression problem because the model predicts a numeric demand
value.

## Saved Models

The trained models are saved in the `models/` folder:

- `models/lag_blend_baseline.joblib`
- `models/poisson_regression.joblib`
- `models/hist_gradient_boosting.joblib`

## Evaluation Setup

The project uses a chronological train/test split:

- Train: `2012-01-01` to `2014-06-30`
- Test: `2014-07-01` to `2014-10-31`

This avoids lookahead leakage. A random split would be invalid for forecasting
because it would mix past and future dates.

## Model 1 - Lag Blend Baseline

The baseline is a transparent rule-based model based on recent historical sales.

It predicts demand with a weighted average:

```text
prediction = 45% * lag_7
           + 35% * rolling_mean_28
           + 20% * rolling_mean_90
```

If all historical signals are zero, it falls back to the global mean from the
training set.

This model is important because a machine learning model should beat a simple
business heuristic before its extra complexity is justified.

## Model 2 - Weighted Poisson Regression

Poisson regression is included because `units` is a nonnegative count target.

The model uses:

- one-hot encoding for categorical variables
- scaling for numeric variables
- sample weights to account for zero-row downsampling

It is interpretable and statistically reasonable for count data, but it is
mostly linear. This makes it weaker when demand depends on nonlinear
interactions between products, stores, calendar effects, and weather.

## Model 3 - Histogram Gradient Boosting

Histogram Gradient Boosting is the final selected model.

It is well suited to this problem because:

- the data is tabular
- the relationships are nonlinear
- demand depends on interactions between store, item, calendar, sales history,
  and weather
- it can use categorical features through deterministic categorical dtypes
- it is more powerful than a linear model while remaining explainable enough for
  a proof of concept

The model is trained on `log1p(units)` and predictions are transformed back to
unit sales. This helps with the skewed target distribution.

## Sparse Target Handling

The dataset has many zero-sales rows, about `97.4%` of all observations.

During training:

- all positive-sales rows are kept
- a controlled sample of zero-sales rows is kept
- Poisson regression receives sample weights to better reflect the original
  zero/positive prevalence

Evaluation is always performed on the full chronological test set.

## Metrics

The project reports:

- MAE: average absolute unit error
- RMSE: penalizes large errors more strongly
- RMSLE: focuses on relative error, useful for skewed demand
- R2: share of variance explained
- positive-sales MAE: error only when sales actually happened
- zero-sales MAE: error on rows with zero demand

Positive-sales MAE is important because the global MAE can look good when most
rows have zero sales.

## Model Comparison

Chronological test period: `2014-07-01` to `2014-10-31`

| Model | MAE | RMSE | RMSLE | R2 | Positive-sales MAE | Zero-sales MAE |
|---|---:|---:|---:|---:|---:|---:|
| Histogram Gradient Boosting | 0.222 | 2.404 | 0.093 | 0.893 | 9.261 | 0.009 |
| Weighted Poisson Regression | 1.404 | 6.598 | 0.623 | 0.197 | 29.725 | 0.738 |
| Lag Blend Baseline | 1.238 | 2.960 | 0.694 | 0.838 | 11.086 | 1.006 |

## Selected Model

The selected model is Histogram Gradient Boosting.

It wins because it has:

- the lowest MAE
- the lowest RMSE
- the lowest RMSLE
- the highest R2
- the best positive-sales MAE
- the best zero-sales MAE

The result shows that the nonlinear model improves over both the simple lag
baseline and the count-regression model.

## Interpretation

The strongest signal comes from recent sales history, especially rolling demand
features. Calendar and store/product identifiers also matter.

Weather features are included because they are plausible business context, but
current feature importance suggests they are much weaker than historical demand.
A future ablation test could quantify the exact added value of weather by
training the same model with and without weather variables.
