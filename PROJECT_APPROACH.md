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

A senior review also found that the raw `EMA_10`, `EMA_20`, `EMA_50`, and `EMA_200` columns are future-looking. They satisfy a backward EMA recurrence using `EMA[t + 1]`, so they cannot be used as features.

These columns will be excluded:

```text
EMA_10, EMA_20, EMA_50, EMA_200,
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
trailing_ema_10 = Price.ewm(span=10, adjust=False).mean()
trailing_ema_20 = Price.ewm(span=20, adjust=False).mean()
trailing_ema_50 = Price.ewm(span=50, adjust=False).mean()
trailing_ema_200 = Price.ewm(span=200, adjust=False).mean()
price_vs_trailing_ema_10 = Price / trailing_ema_10 - 1
price_vs_trailing_ema_20 = Price / trailing_ema_20 - 1
price_vs_trailing_ema_50 = Price / trailing_ema_50 - 1
price_vs_trailing_ema_200 = Price / trailing_ema_200 - 1
```

These features are valid because they use information up to the current row only.

To avoid same-close execution bias, all feature columns are shifted by one trading day within each index before modeling. This means the model uses information available through `t - 1` to decide whether to be long from `t` to `t + 1`.

Additional implemented features:

```python
volatility_ratio_5d_20d = volatility_5d / volatility_20d
return_5d_to_volatility_20d = return_5d / volatility_20d
return_20d_to_volatility_20d = return_20d / volatility_20d
drawdown_20d = Price / rolling_20d_high - 1
drawdown_60d = Price / rolling_60d_high - 1
price_vs_20d_low = Price / rolling_20d_low - 1
trailing_ema_10_vs_50 = trailing_ema_10 / trailing_ema_50 - 1
trailing_ema_20_vs_200 = trailing_ema_20 / trailing_ema_200 - 1
```

These features capture volatility regime, risk-adjusted momentum, trend state, and drawdown/recovery state without using future information.

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
EMA_10
EMA_20
EMA_50
EMA_200
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

## 8. Modeling Plan And Current Results

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

### Model 3: Hist Gradient Boosting Classifier

Purpose:

- Strong supervised tabular model.
- Often performs well on structured financial datasets.
- Can capture nonlinear interactions across macro, futures, rates, and cross-market variables.

Expected role:

- Main performance candidate.
- Likely best model for predictive and trading metrics.

### Current Classification Results

The implemented models are:

- Logistic Regression
- Random Forest
- Hist Gradient Boosting

Saved models:

```text
models/log_reg.joblib
models/random_forest.joblib
models/hist_gradient_boosting.joblib
```

Current registered test metrics from `results/model_metrics.csv`:

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.4978 | 0.5010 | 0.4911 | 0.6692 | 0.5665 |
| Random Forest | 0.5186 | 0.5218 | 0.5067 | 0.6859 | 0.5828 |
| Hist Gradient Boosting | 0.5089 | 0.5145 | 0.4995 | 0.8027 | 0.6158 |

The corrected classifiers are close to chance. Random Forest has the best registered test accuracy, but none of the models is strong enough to claim a robust predictive edge from classification metrics alone.

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

The implemented first version includes a 1 basis point cost per signal change.

The strategy is evaluated as an equal-weight daily portfolio across the three index signals. This avoids overstating annualization from treating the three index rows as separate trading days.

## 10. Evaluation Metrics

### Classification Metrics

The metrics function in `src/metrics.py` computes:

```text
accuracy
balanced_accuracy
precision
recall
f1
roc_auc
```

ROC-AUC is included when predicted probabilities are available. The template-compatible
`compute_metrics()` wrapper still returns the core class-label metrics.

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

## 11. Current Results And Interpretation

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

### Current Backtest Results

Backtest outputs:

```text
results/strategy_metrics.csv
results/equity_curves.csv
results/strategy_returns.csv
```

The model strategy:

1. Trains a temporary model on 2010-2019.
2. Selects the long/cash probability threshold on 2020-2021 validation data.
3. Applies that threshold to the final saved model on the 2022-2023 test period.
4. Charges 1 basis point per signal change.

The original high-return result was invalidated because it depended on the dataset's future-looking `EMA_*` columns. After removing those columns, recomputing trailing EMAs, and lagging all features by one day, the corrected test-period results are:

| Strategy | Threshold | Total return | Annualized return | Sharpe | Max drawdown | Exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Buy and hold | n/a | -0.1014 | -0.0411 | -0.2102 | -0.2549 | 1.0000 |
| Previous-day direction | n/a | -0.2518 | -0.1552 | -1.2371 | -0.2697 | 0.4896 |
| Price above trailing EMA 20 | n/a | -0.1097 | -0.0596 | -0.5558 | -0.1692 | 0.4926 |
| Positive 20d momentum | n/a | -0.1226 | -0.0670 | -0.5858 | -0.1874 | 0.4807 |
| Logistic Regression long/cash | 0.60 | 0.0966 | 0.0572 | 0.5531 | -0.0902 | 0.2552 |
| Random Forest long/cash | 0.60 | -0.0006 | 0.0020 | 0.0294 | -0.0645 | 0.0915 |
| Hist Gradient Boosting long/cash | 0.60 | -0.0640 | -0.0293 | -0.2326 | -0.1486 | 0.3854 |

These corrected results are much more realistic. Logistic Regression improves over buy-and-hold during a difficult test period, but the edge is modest and needs further validation before being presented as robust.

Important next stress tests:

- Per-index strategy metrics.
- Higher transaction cost sensitivity.
- Walk-forward validation instead of a single train/validation/test split.
- Probability calibration check.

## 12. Implementation Status

### Step 1: Data Loading - Complete

Implemented in `src/data.py`:

1. Load the three CSV files.
2. Add an `index_name` column from the file name or `Name`.
3. Sort each index by date.
4. Create `next_return` and `target`.
5. Drop or exclude leakage columns.
6. Engineer non-leaky trailing features.
7. Combine the three files into one modeling dataset.
8. Use a chronological split.

The returned values follow the template contract:

```python
return X_train, X_test, y_train, y_test
```

Processed files generated by `scripts/prepare_data.py`:

```text
data/processed_modeling_dataset.csv
data/processed_train.csv
data/processed_validation.csv
data/processed_test.csv
```

Processed dataset summary:

| File | Rows | Columns |
| --- | ---: | ---: |
| `processed_modeling_dataset.csv` | 10,227 | 100 |
| `processed_train.csv` | 7,368 | 100 |
| `processed_validation.csv` | 1,515 | 100 |
| `processed_test.csv` | 1,344 | 100 |

### Step 2: Training Script - Complete

Implemented:

```text
scripts/train_models.py
```

This script:

1. Load the modeling dataset.
2. Fit preprocessing pipelines.
3. Train Logistic Regression, Random Forest, and Hist Gradient Boosting.
4. Save models into `models/`.
5. Save validation/test classification results.

### Step 3: Model Registry - Complete

Implemented in `src/config.py`:

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
    "hist_gradient_boosting": {
        "name": "Hist Gradient Boosting",
        "description": "Gradient boosted trees for tabular market features.",
        "path": MODELS_DIR / "hist_gradient_boosting.joblib",
    },
}
```

### Step 4: Metrics - Complete

Implemented in `src/metrics.py`:

```text
accuracy
balanced_accuracy
precision
recall
f1
optional roc_auc
```

Use stable metric names so the CSV output remains consistent across models.

### Step 5: Backtest Evaluation - Complete

Implemented:

```text
src/backtest.py
scripts/run_backtest.py
```

These modules:

1. Convert model probabilities into signals.
2. Calculate strategy returns.
3. Compare against buy-and-hold and simple baselines.
4. Save strategy outputs to `results/`.

### Step 6: Streamlit App - Remaining

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

### Step 7: Tests And Tooling - Complete

Implemented:

```text
pyproject.toml
setup.cfg
requirements-dev.txt
tests/test_data_pipeline.py
tests/test_metrics.py
tests/test_backtest.py
```

The tests cover the highest-risk parts of the project:

1. Leakage columns and raw future-looking EMA columns are excluded from features.
2. `next_return` and `target` match the next trading day's price move by index.
3. Price-derived features are lagged one trading day before modeling.
4. Shared classification metrics behave consistently.
5. Backtest signal, transaction cost, portfolio aggregation, and threshold logic work.

Quality checks currently pass:

```text
python -m pytest
python -m compileall src scripts tests
python -m black --check src scripts tests
python -m flake8 src scripts tests
python -m pylint src scripts tests
```

## 13. Repo Deliverables

The project currently produces:

```text
data/
  combined_dataframe_DJI.csv
  combined_dataframe_IXIC.csv
  combined_dataframe_NYSE.csv
  processed_modeling_dataset.csv
  processed_train.csv
  processed_validation.csv
  processed_test.csv

models/
  log_reg.joblib
  random_forest.joblib
  hist_gradient_boosting.joblib

results/
  model_metrics.csv
  strategy_metrics.csv
  equity_curves.csv
  strategy_returns.csv
  training_classification_report.csv

plots/
  test_equity_curves.png
  test_drawdowns.png
  test_strategy_metric_bars.png
  test_model_metric_bars.png
  test_return_distribution.png
  test_model_exposure.png

src/
  data.py
  metrics.py
  modeling.py
  backtest.py
  app.py
  config.py

scripts/
  prepare_data.py
  train_models.py
  run_backtest.py
  generate_plots.py
  main.py

tests/
  test_data_pipeline.py
  test_metrics.py
  test_backtest.py

pyproject.toml
setup.cfg
requirements-dev.txt
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
