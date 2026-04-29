# Quant Finance ML POC Approach

## 1. Project Objective

This project will build a supervised machine learning proof of concept for short-horizon equity index prediction.

The goal is not to forecast the exact future level of an index. Daily index prices are noisy and close to efficient, so direct price regression is unlikely to be convincing. Instead, the project will focus on a more realistic quant-finance task:

> Predict whether a major US equity index will produce a positive next-day return, then convert the model probability into a simple long/cash trading signal.

The project will evaluate success in two ways:

1. Predictive performance: accuracy, F1 score, ROC-AUC, precision, recall, and confusion matrix.
2. Trading performance: annualized return, annualized volatility, Sharpe ratio, max drawdown, turnover, and comparison against baseline strategies.

This creates a defensible supervised learning project while keeping the quant-finance angle explicit.

## 2. Dataset Overview

The added data consists of three CSV files in `data/`:

| File | Index | Rows | Columns | Date range |
| --- | --- | ---: | ---: | --- |
| `combined_dataframe_DJI.csv` | Dow Jones Industrial Average | 3,470 | 84 | 2010-01-04 to 2023-10-16 |
| `combined_dataframe_IXIC.csv` | NASDAQ Composite | 3,470 | 84 | 2010-01-04 to 2023-10-16 |
| `combined_dataframe_NYSE.csv` | NYSE Composite | 3,470 | 84 | 2010-01-04 to 2023-10-16 |

Each row is a trading day. The main index level is stored in `Price`, and the file-specific index identifier is stored in `Name`.

The columns include several groups of explanatory variables:

- Calendar feature: `weekday`
- Index price features: `Price`, `EMA_10`, `EMA_20`, `EMA_50`, `EMA_200`
- Cross-market equity variables: `GSPC`, `IXIC`, `DJI`, `NYSE`, `RUT`, `FTSE`, `GDAXI`, `FCHI`, `HSI`, `SSEC`
- Futures variables: `S&P-F`, `NASDAQ-F`, `DJI-F`, `RUSSELL-F`, `FTSE-F`, `DAX-F`, `CAC-F`, `Nikkei-F`, `KOSPI-F`
- Commodity and FX variables: oil, Brent, WTI oil, gold, silver, copper, wheat, gas, dollar index, EUR, GBP, JPY, CHF, CAD, AUD, NZD, CNY
- Rates and credit variables: `DGS10`, `DGS5`, `DTB3`, `DTB4WK`, `DTB6`, `CTB3M`, `CTB6M`, `CTB1Y`, `DAAA`, `DBAA`
- Yield/spread features: `TE1`, `TE2`, `TE3`, `TE5`, `TE6`, `DE1`, `DE2`, `DE4`, `DE5`, `DE6`
- Large-cap stock variables: `AAPL`, `AMZN`, `MSFT`, `JPM`, `JNJ`, `XOM`, `GE`, `WFC`

This is a useful dataset because it is not limited to one price series. It gives the models macro, cross-asset, cross-index, futures, FX, commodity, credit, and stock-level context.

## 3. Important Leakage Audit

The dataset contains columns that appear to use future index prices. These must not be used as model inputs.

The strongest example is `mom`. For the first DJI row:

- `Price` on 2010-01-04 is 10,583.96
- `Price` on 2010-01-05 is 10,572.02
- The inverse next-day return is `10583.96 / 10572.02 - 1 = 0.001129`
- The dataset's `mom` value is also `0.001129`

So `mom` encodes the next day's movement. That is target leakage.

The `ROC_5`, `ROC_10`, `ROC_15`, and `ROC_20` columns also appear to be forward-looking. For example, `ROC_5` matches the inverse five-day forward return.

These columns will be excluded:

```text
mom, mom1, mom2, mom3, ROC_5, ROC_10, ROC_15, ROC_20
```

This is important because a model using these columns would show unrealistically strong results. The project should explicitly mention this leakage audit because it demonstrates good modeling discipline.

## 4. Target Definition

The main supervised target will be next-day direction:

```python
next_return_t = Price[t + 1] / Price[t] - 1
target_t = 1 if next_return_t > 0 else 0
```

The model will use information available at day `t` to predict the sign of day `t + 1`.

This gives a binary classification problem:

- `1`: next-day return is positive
- `0`: next-day return is zero or negative

The trading interpretation is:

- If the model is sufficiently confident that `target = 1`, go long the index for the next day.
- Otherwise, stay in cash.

We may also build a secondary regression target:

```python
next_return_t = Price[t + 1] / Price[t] - 1
```

However, the primary project should be classification because directional prediction is easier to explain, easier to evaluate, and more directly connected to the long/cash strategy.

## 5. Initial Dataset Findings

The class balance is usable. The full-sample next-day up rates are:

| Index | Next-day up rate |
| --- | ---: |
| DJI | 54.1% |
| IXIC | 55.5% |
| NYSE | 53.5% |

The data covers several market regimes:

- 2010-2019: long post-crisis bull market
- 2020-2021: COVID crash and recovery
- 2022-2023: higher-rate drawdown/recovery period

This is useful for testing whether a model generalizes across regimes.

The proposed chronological split is:

| Split | Dates | Rows per index | Purpose |
| --- | --- | ---: | --- |
| Train | 2010-2019 | 2,516 | Fit models |
| Validation | 2020-2021 | 505 | Tune thresholds and hyperparameters |
| Test | 2022-2023 | 449 | Final out-of-sample evaluation |

We should not use a random train/test split because that would mix future observations into the training set and overstate real-world performance.

## 6. Baselines

The project needs strong baselines so that any model result is meaningful.

### Predictive Baselines

1. Majority class baseline
   - Always predict that the next day is up.
   - This is hard to beat on accuracy because the market has an upward drift.

2. Previous-day direction baseline
   - Predict tomorrow will have the same direction as today.

3. Simple logistic baseline
   - Use a small set of non-leaky features such as lagged returns, rolling volatility, weekday, and price-vs-moving-average features.

### Trading Baselines

1. Buy and hold
   - Stay long the index every day.

2. Cash
   - No exposure.

3. Moving-average rule
   - Long when price is above a moving average, otherwise cash.

4. Naive momentum rule
   - Long if recent non-leaky trailing return is positive.

The model should be judged against these baselines using test-period results, not only full-sample results.

## 7. Feature Engineering Plan

The project should avoid relying only on the provided columns. We will create clean, non-leaky features from `Price` and from the available market variables.

### Non-Leaky Index Features

For each index file:

```python
return_1d = Price.pct_change(1)
return_2d = Price.pct_change(2)
return_5d = Price.pct_change(5)
return_10d = Price.pct_change(10)
return_20d = Price.pct_change(20)
volatility_5d = return_1d.rolling(5).std()
volatility_10d = return_1d.rolling(10).std()
volatility_20d = return_1d.rolling(20).std()
price_vs_ema_10 = Price / EMA_10 - 1
price_vs_ema_20 = Price / EMA_20 - 1
price_vs_ema_50 = Price / EMA_50 - 1
price_vs_ema_200 = Price / EMA_200 - 1
```

These features are valid because they use information up to the current row only.

### Existing Market Features

We will keep the non-leaky cross-market, macro, FX, commodity, rates, futures, and stock variables.

Missing values will be handled carefully:

- Sort by date.
- Forward-fill market variables where appropriate.
- Back-fill only after the train/test split logic is clear, or avoid back-fill entirely if it could introduce future information.
- Drop the earliest rows where rolling features are not available.
- Keep missingness handling inside a scikit-learn pipeline where possible.

### Excluded Features

The following columns will not be used:

```text
Date
Name
Price as a raw level
mom
mom1
mom2
mom3
ROC_5
ROC_10
ROC_15
ROC_20
```

Raw `Price` will not be used directly because it is non-stationary. Instead, it will be converted into returns, rolling volatility, and relative-to-moving-average features.

## 8. Modeling Plan

All models will be supervised.

### Model 1: Logistic Regression

Purpose:

- Transparent baseline.
- Works well with standardized numeric features.
- Produces probabilities for threshold-based trading.

Expected role:

- Main interpretable benchmark.
- Helps determine whether the signal is real or only captured by complex models.

### Model 2: Random Forest Classifier

Purpose:

- Captures nonlinear relationships and feature interactions.
- Robust to mixed feature scales.

Expected role:

- Tree-based benchmark.
- Useful for feature importance and nonlinear signal discovery.

### Model 3: Gradient Boosting / XGBoost Classifier

Purpose:

- Strong supervised tabular model.
- Often performs well on structured financial datasets.
- Can capture nonlinear interactions across macro, futures, rates, and cross-market variables.

Expected role:

- Main performance candidate.
- Likely best model for predictive and trading metrics.

### Optional Model 4: Calibrated Classifier

If probability quality matters, we can calibrate the best classifier using validation data.

Purpose:

- Make predicted probabilities more reliable.
- Improve threshold selection for the trading strategy.

This is useful because the trading signal depends on `P(up)`, not just the hard class prediction.

## 9. Trading Rule

The model will output a probability:

```python
p_up = model.predict_proba(X)[1]
```

The default trading rule will be:

```python
signal = 1 if p_up > threshold else 0
strategy_return = signal * next_return
```

The threshold will be tuned on the validation period only.

Candidate thresholds:

```text
0.50, 0.52, 0.55, 0.57, 0.60
```

Using a threshold above 0.50 is important. It allows the strategy to avoid low-confidence days, which can improve Sharpe even if raw classification accuracy is only modest.

We will also report turnover:

```python
turnover = average(abs(signal[t] - signal[t - 1]))
```

If needed, we can subtract simple transaction costs:

```python
net_strategy_return = signal * next_return - cost_per_trade * abs(signal[t] - signal[t - 1])
```

For the first version, use no transaction costs and then add a sensitivity check.

## 10. Evaluation Metrics

### Classification Metrics

The metrics function in `src/metrics.py` should compute:

```text
accuracy
precision
recall
f1
roc_auc
```

ROC-AUC requires probabilities, not just class labels. The current template only passes `y_pred` into `compute_metrics`, so we have two options:

1. Keep `src/metrics.py` simple and compute accuracy, precision, recall, and F1 only.
2. Extend the orchestration later to also pass predicted probabilities and support ROC-AUC.

For the first clean implementation, use:

```text
accuracy
precision
recall
f1
```

Then add ROC-AUC in a later iteration if we adjust the template.

### Strategy Metrics

Strategy metrics should be calculated separately from `compute_metrics`, because they require dates and next-day returns.

We will compute:

```text
annualized_return = mean(daily_returns) * 252
annualized_volatility = std(daily_returns) * sqrt(252)
sharpe_ratio = annualized_return / annualized_volatility
max_drawdown
win_rate
exposure
turnover
```

The Streamlit app should show classification metrics and trading metrics side by side.

## 11. Expected Results

We should set realistic expectations.

Daily market direction is difficult. A good model may only achieve 52-56% directional accuracy out of sample. That can still be useful if the model improves risk-adjusted returns by avoiding weak or negative regimes.

The strongest claim should not be:

> We can predict the stock market accurately.

The stronger and more defensible claim is:

> Cross-market and macro-financial features can improve a supervised long/cash index timing strategy relative to naive baselines during the test period.

The final test period, 2022-2023, is especially useful because buy-and-hold was weaker for all three indices:

| Index | Test-period buy-and-hold annualized return | Test-period buy-and-hold Sharpe |
| --- | ---: | ---: |
| DJI | -2.8% | -0.17 |
| IXIC | -5.1% | -0.19 |
| NYSE | -4.4% | -0.25 |

This gives the model a meaningful challenge: can it reduce exposure during a more difficult market regime?

## 12. Implementation Order

### Step 1: Data Loading

Update `src/data.py` to:

1. Load the three CSV files.
2. Add an `index_name` column from the file name or `Name`.
3. Sort each index by date.
4. Create `next_return` and `target`.
5. Drop or exclude leakage columns.
6. Engineer non-leaky trailing features.
7. Combine the three files into one modeling dataset.
8. Use a chronological split.

The returned values should follow the template contract:

```python
return X_train, X_test, y_train, y_test
```

For the first implementation, validation can be handled inside the training script or by using the 2020-2021 period before final testing.

### Step 2: Training Script

Add a training script, for example:

```text
scripts/train_models.py
```

This script should:

1. Load the modeling dataset.
2. Fit preprocessing pipelines.
3. Train Logistic Regression, Random Forest, and XGBoost or Gradient Boosting.
4. Save models into `models/`.
5. Save any feature list or preprocessing metadata needed by the app.

### Step 3: Model Registry

Update `src/config.py`:

```python
MODELS = {
    "log_reg": {
        "name": "Logistic Regression",
        "description": "Linear baseline with standardized features.",
        "path": MODELS_DIR / "log_reg.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Nonlinear tree ensemble benchmark.",
        "path": MODELS_DIR / "random_forest.joblib",
    },
    "xgboost": {
        "name": "XGBoost",
        "description": "Gradient boosted trees for tabular market features.",
        "path": MODELS_DIR / "xgboost.joblib",
    },
}
```

If XGBoost is too heavy or unavailable, use scikit-learn's `HistGradientBoostingClassifier`.

### Step 4: Metrics

Update `src/metrics.py` to compute classification metrics:

```text
accuracy
precision
recall
f1
```

Use stable metric names so the CSV output remains consistent across models.

### Step 5: Backtest Evaluation

Add a utility module, for example:

```text
src/backtest.py
```

This module should:

1. Convert model probabilities into signals.
2. Calculate strategy returns.
3. Compare against buy-and-hold and simple baselines.
4. Save strategy outputs to `results/strategy_metrics.csv`.

### Step 6: Streamlit App

Customize `src/app.py` to show:

1. Project objective.
2. Dataset summary and leakage audit.
3. Train/validation/test split.
4. Model comparison table.
5. Strategy metrics table.
6. Equity curve chart.
7. Drawdown chart.
8. Feature importance for tree models.
9. Interactive controls:
   - Select index: DJI, IXIC, NYSE
   - Select model
   - Select probability threshold
   - Include or exclude transaction costs

The app should focus on the actual quant workflow, not a generic ML dashboard.

## 13. Repo Deliverables

The final project should produce:

```text
data/
  combined_dataframe_DJI.csv
  combined_dataframe_IXIC.csv
  combined_dataframe_NYSE.csv

models/
  log_reg.joblib
  random_forest.joblib
  xgboost.joblib

results/
  model_metrics.csv
  strategy_metrics.csv
  equity_curves.csv

plots/
  optional saved charts

src/
  data.py
  metrics.py
  backtest.py
  app.py
  config.py

scripts/
  train_models.py
  main.py
```

## 14. Success Criteria

The project will be considered successful if it satisfies these conditions:

1. Uses only supervised models.
2. Uses a chronological split, not a random split.
3. Explicitly removes leakage columns.
4. Beats at least one meaningful predictive baseline on the test period.
5. Beats buy-and-hold or improves drawdown/Sharpe during the test period for at least one index.
6. Shows both classification metrics and trading metrics.
7. Provides a Streamlit app that explains the dataset, model, results, and trading interpretation clearly.

The highest-quality version would show that a model may not predict every day well, but can still improve risk-adjusted exposure by trading only when the predicted probability is strong enough.

