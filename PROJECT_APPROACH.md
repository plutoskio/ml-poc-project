# Quant Finance ML POC: Full Project Methodology

## 1. Project Objective

This project builds a supervised machine learning proof of concept for short-horizon equity index prediction.

The goal is not to predict the exact future price of an index. Daily index levels are noisy, non-stationary, and difficult to forecast directly. Instead, we framed the problem as a supervised classification task:

```python
target = 1 if next_return > 0 else 0
```

The model predicts whether the next trading day is positive or negative. We then convert the predicted probability into a simple long/cash trading strategy and compare it against baseline strategies.

The final project evaluates two things:

1. Predictive performance: accuracy, balanced accuracy, precision, recall, and F1.
2. Trading performance: total return, annualized return, volatility, Sharpe ratio, max drawdown, win rate, exposure, and turnover.

The strongest defensible claim is not that we can predict the stock market every day. The claim is that a supervised model can sometimes improve risk-adjusted exposure by trading only on higher-confidence days.

## 2. Data Used

The project uses three daily US equity index datasets:

| File | Index | Rows | Date range |
| --- | --- | ---: | --- |
| `data/combined_dataframe_DJI.csv` | Dow Jones Industrial Average | 3,470 | 2010-01-04 to 2023-10-16 |
| `data/combined_dataframe_IXIC.csv` | NASDAQ Composite | 3,470 | 2010-01-04 to 2023-10-16 |
| `data/combined_dataframe_NYSE.csv` | NYSE Composite | 3,470 | 2010-01-04 to 2023-10-16 |

Each row is one trading day. The main index level is stored in `Price`.

The raw data includes:

- Index prices and technical indicators.
- Cross-index variables such as `GSPC`, `IXIC`, `DJI`, `NYSE`, and `RUT`.
- Futures variables such as `S&P-F`, `NASDAQ-F`, `DJI-F`, and international futures.
- Commodity and FX variables such as oil, gold, silver, dollar index, EUR, GBP, JPY, and others.
- Rates and credit variables such as `DGS10`, `DGS5`, `DTB3`, `DAAA`, and `DBAA`.
- Large-cap stock variables such as `AAPL`, `AMZN`, `MSFT`, `JPM`, `JNJ`, `XOM`, `GE`, and `WFC`.

This dataset is useful because it gives the model broader market context instead of only using one price series.

## 3. Leakage Audit

The most important part of the project was checking for leakage. Several raw columns were found to contain future information and were removed.

### Removed Momentum And ROC Columns

The following columns were excluded:

```text
mom, mom1, mom2, mom3, ROC_5, ROC_10, ROC_15, ROC_20
```

Reason: these columns appeared to encode forward returns. For example, `mom` matched the inverse next-day return in the raw data. Using it would allow the model to indirectly see the answer.

### Removed Raw EMA Columns

The following raw EMA columns were also excluded:

```text
EMA_10, EMA_20, EMA_50, EMA_200
```

Reason: a senior review found that the dataset-provided EMA columns were future-looking. They followed a backward recurrence using future EMA values, so they were not valid features at time `t`.

### Same-Close Timing Bias Fix

We also avoided using same-day close-derived features to trade the same close-to-close return.

All model feature columns are shifted by one trading day within each index. This means the model uses information available through the previous trading day to decide whether to be invested for the next return.

This makes the backtest more conservative and avoids execution timing bias.

## 4. Target Creation

For each index separately, we sorted rows by date and created:

```python
next_return_t = Price[t + 1] / Price[t] - 1
target_t = 1 if next_return_t > 0 else 0
```

Interpretation:

- `target = 1`: the next trading day was positive.
- `target = 0`: the next trading day was flat or negative.

The final row for each index has no next-day return, so it is dropped from supervised training/evaluation.

## 5. Feature Engineering

The pipeline keeps non-leaky raw numeric market variables and creates additional clean price-derived features.

Engineered features include:

```python
return_1d
return_2d
return_5d
return_10d
return_20d
volatility_5d
volatility_10d
volatility_20d
volatility_ratio_5d_20d
return_5d_to_volatility_20d
return_20d_to_volatility_20d
drawdown_20d
drawdown_60d
price_vs_20d_low
price_vs_trailing_ema_10
price_vs_trailing_ema_20
price_vs_trailing_ema_50
price_vs_trailing_ema_200
trailing_ema_10_vs_50
trailing_ema_20_vs_200
```

Important implementation details:

- Raw `Price` is not used directly as a model feature because it is non-stationary.
- Valid trailing EMAs are recomputed from `Price` using past data only.
- Features are forward-filled within each index using only current/past values.
- After feature creation, all numeric model features are shifted by one trading day.
- Remaining missing values are filled with training-period medians, then zero if needed.
- Index dummy variables are added: `index_DJI`, `index_IXIC`, and `index_NYSE`.

The final feature matrix contains 93 model input columns.

## 6. Chronological Split

The project uses a time-based split, not a random split.

| Split | Dates | Rows | Purpose |
| --- | --- | ---: | --- |
| Train | 2010-01-04 to 2019-12-31 | 7,368 | Fit models |
| Validation | 2020-01-01 to 2021-12-31 | 1,515 | Select trading thresholds |
| Test | 2022-01-01 to 2023-10-16 | 1,344 | Final out-of-sample evaluation |

Why this matters:

- Random splitting would mix future observations into the training set.
- Finance models must be evaluated by training on the past and testing on the future.
- The 2022-2023 test period is a useful challenge because buy-and-hold was weak.

Saved processed files:

```text
data/processed_modeling_dataset.csv
data/processed_train.csv
data/processed_validation.csv
data/processed_test.csv
```

## 7. Models Trained

All models are supervised classifiers.

### Logistic Regression

Purpose:

- Transparent baseline.
- Produces probabilities.
- Useful for checking whether a simple linear model can find signal.

### Random Forest

Purpose:

- Nonlinear tree benchmark.
- Captures feature interactions.
- More flexible than Logistic Regression.

### Hist Gradient Boosting

Purpose:

- Strong tabular supervised model.
- Captures nonlinear relationships across market variables.
- Used as the main complex model candidate.

Saved model files:

```text
models/log_reg.joblib
models/random_forest.joblib
models/hist_gradient_boosting.joblib
```

## 8. Classification Evaluation

The template evaluation flow loads each saved model, predicts on the test set, and writes:

```text
results/model_metrics.csv
```

Current corrected test classification results:

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.4978 | 0.5010 | 0.4911 | 0.6692 | 0.5665 |
| Random Forest | 0.5186 | 0.5218 | 0.5067 | 0.6859 | 0.5828 |
| Hist Gradient Boosting | 0.5089 | 0.5145 | 0.4995 | 0.8027 | 0.6158 |

Interpretation:

- The models are close to chance on pure next-day direction.
- Random Forest has the best test accuracy, but the edge is small.
- This is realistic for daily index prediction.
- Classification accuracy alone is not enough, so we also evaluate trading performance.

## 9. Trading Strategy

The model strategy is a long/cash strategy.

For each date and index:

```python
p_up = model probability that next_return > 0
signal = 1 if p_up >= threshold else 0
```

Signal meaning:

- `signal = 1`: go long the index for the next return.
- `signal = 0`: stay in cash.

The strategy return is:

```python
gross_strategy_return = signal * next_return
strategy_return = gross_strategy_return - cost_per_trade * trade
```

Where:

```python
trade = abs(signal[t] - signal[t - 1])
cost_per_trade = 0.0001
```

So the model does not trade every day. It only has exposure when its probability is above the selected threshold. If the signal stays long across consecutive days, it remains invested and does not pay a new entry cost every day.

Daily portfolio returns are aggregated as an equal-weight portfolio across the three index signals. This avoids treating three index rows as three independent trading days.

## 10. Validation Threshold Selection

The threshold is selected on the validation period only.

Candidate thresholds:

```text
0.50, 0.52, 0.55, 0.57, 0.60
```

Process:

1. Train a temporary model on the train split only: 2010-2019.
2. Generate validation probabilities for 2020-2021.
3. Backtest each candidate threshold on validation.
4. Select the threshold with the best validation Sharpe ratio, requiring at least 10% exposure.
5. Apply the selected threshold to the untouched test period.

Current selected threshold:

```text
0.60 for all three model strategies
```

This means the final model strategy only invests when the model predicts at least a 60% probability of an up day.

## 11. Baselines

The model is compared against both prediction and trading baselines.

Trading baselines:

- `buy_and_hold`: always long.
- `cash`: never invested.
- `previous_day_direction`: long if the previous day's lagged return was positive.
- `price_above_trailing_ema_20`: long when price is above trailing EMA 20.
- `positive_20d_momentum`: long when trailing 20-day return is positive.

These baselines matter because the model should not only look good in isolation. It should improve over simple alternatives.

## 12. Corrected Backtest Results

The first high-return result was rejected after the leakage review because it depended on invalid future-looking EMA features.

After removing leakage columns, recomputing valid trailing EMAs, and lagging features by one trading day, the corrected test results are:

| Strategy | Threshold | Total return | Annualized return | Sharpe | Max drawdown | Exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Buy and hold | n/a | -0.1014 | -0.0411 | -0.2102 | -0.2549 | 1.0000 |
| Previous-day direction | n/a | -0.2518 | -0.1552 | -1.2371 | -0.2697 | 0.4896 |
| Price above trailing EMA 20 | n/a | -0.1097 | -0.0596 | -0.5558 | -0.1692 | 0.4926 |
| Positive 20d momentum | n/a | -0.1226 | -0.0670 | -0.5858 | -0.1874 | 0.4807 |
| Logistic Regression long/cash | 0.60 | 0.0966 | 0.0572 | 0.5531 | -0.0902 | 0.2552 |
| Random Forest long/cash | 0.60 | -0.0006 | 0.0020 | 0.0294 | -0.0645 | 0.0915 |
| Hist Gradient Boosting long/cash | 0.60 | -0.0640 | -0.0293 | -0.2326 | -0.1486 | 0.3854 |

Interpretation:

- Buy-and-hold lost about 10.1% in the test period.
- Logistic Regression produced a positive 9.7% total return with lower drawdown and only 25.5% exposure.
- Random Forest was roughly flat.
- Hist Gradient Boosting lost money but still had lower drawdown than buy-and-hold.
- The best corrected result is Logistic Regression, but the edge is modest and should not be overclaimed.

## 13. Plot Outputs

The project generates visual outputs in `plots/`:

```text
plots/test_equity_curves.png
plots/test_drawdowns.png
plots/test_strategy_metric_bars.png
plots/test_model_metric_bars.png
plots/test_return_distribution.png
plots/test_model_exposure.png
```

How to read the key plots:

- `test_equity_curves.png`: growth of $1 for model strategies and baselines.
- `test_drawdowns.png`: peak-to-trough declines through time.
- `test_strategy_metric_bars.png`: total return, Sharpe, and max drawdown comparison.
- `test_model_metric_bars.png`: classification metrics by model.
- `test_return_distribution.png`: daily return distribution comparison.
- `test_model_exposure.png`: fraction of test days each model strategy was invested.

For exposure, a value of 0.255 means the strategy was long about 25.5% of test days and in cash the rest of the time.

## 14. Tests And Quality Checks

Tests were added for the highest-risk parts of the project:

```text
tests/test_data_pipeline.py
tests/test_metrics.py
tests/test_backtest.py
```

The tests verify:

- Leakage columns are excluded from model features.
- Raw future-looking EMA columns are excluded.
- `next_return` and `target` are correctly created by index.
- Price-derived features are lagged one trading day.
- Classification metrics return stable numeric values.
- Backtest signal, cost, threshold, and portfolio aggregation logic work.

Development tooling was added:

```text
pyproject.toml
setup.cfg
requirements-dev.txt
```

Current checks pass:

```bash
python -m pytest
python -m compileall src scripts tests
python -m black --check src scripts tests
python -m flake8 src scripts tests
python -m pylint src scripts tests
```

## 15. Project Files

Main implementation files:

```text
src/data.py              # data loading, target creation, leakage removal, features
src/modeling.py          # model definitions and probability helper
src/metrics.py           # classification metrics
src/backtest.py          # long/cash backtest helpers
src/config.py            # paths and model registry
src/app.py               # Streamlit entry point
scripts/prepare_data.py  # writes processed datasets
scripts/train_models.py  # trains and saves models
scripts/run_backtest.py  # threshold selection and strategy backtests
scripts/generate_plots.py # plot generation
scripts/main.py          # template evaluation and Streamlit launcher
```

Main output files:

```text
data/processed_modeling_dataset.csv
data/processed_train.csv
data/processed_validation.csv
data/processed_test.csv
results/model_metrics.csv
results/training_classification_report.csv
results/strategy_metrics.csv
results/equity_curves.csv
results/strategy_returns.csv
models/log_reg.joblib
models/random_forest.joblib
models/hist_gradient_boosting.joblib
```

## 16. Final Interpretation

The corrected project is realistic:

- It uses only supervised models.
- It uses a chronological split.
- It removes identified future-looking columns.
- It avoids same-close timing bias by lagging features.
- It tunes trading thresholds only on validation.
- It evaluates final performance only on the test period.
- It compares model strategies against simple baselines.

The original high-return result was not valid because of leakage. After fixing the leakage, the models are much closer to chance, which is expected for daily equity index direction.

The most promising result is the Logistic Regression long/cash strategy. It does not predict the market accurately every day, but it selectively invests when predicted confidence is high and produced better test-period risk-adjusted performance than buy-and-hold in 2022-2023.

The result should be presented as a cautious proof of concept, not as a production trading strategy. The next improvements would be walk-forward validation, per-index reporting, transaction-cost sensitivity, probability calibration, and a customized Streamlit app.
