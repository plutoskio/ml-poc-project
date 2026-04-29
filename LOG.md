# Project Log

## 2026-04-29

### Current Direction

We are building a supervised quant-finance ML proof of concept using the three index datasets in `data/`:

- `combined_dataframe_DJI.csv`
- `combined_dataframe_IXIC.csv`
- `combined_dataframe_NYSE.csv`

The project target will be next-day market direction:

```python
next_return = Price[t + 1] / Price[t] - 1
target = 1 if next_return > 0 else 0
```

The model will predict whether the next trading day is up or down, then we will convert model probabilities into a long/cash trading signal and evaluate Sharpe ratio, drawdown, and returns versus baselines.

### Important Data Finding

Some columns appear to leak future information and must be excluded from model inputs:

```text
mom, mom1, mom2, mom3, ROC_5, ROC_10, ROC_15, ROC_20
```

Example: `mom` matches the inverse next-day return, so using it would make the model unrealistically good.

### First Implementation Step

The first step is not the train/test split by itself. The correct order is:

1. Load and sort each index dataset by `Date`.
2. Create `next_return` from `Price[t + 1] / Price[t] - 1`.
3. Create the binary target variable: `target = 1` if `next_return > 0`, else `0`.
4. Remove leakage-prone columns.
5. Engineer only non-leaky features using information available at day `t`.
6. Apply a chronological split:
   - Train: 2010-2019
   - Validation: 2020-2021
   - Test: 2022-2023

This ordering matters because the split must happen on a clean supervised dataset with a valid target.

### Planned Models

- Logistic Regression: transparent baseline.
- Random Forest: nonlinear tree benchmark.
- XGBoost or HistGradientBoosting: main performance candidate.

### Planned Baselines

- Majority class prediction.
- Previous-day direction.
- Buy and hold.
- Moving-average long/cash rule.

### Next Task

Implement the data preparation pipeline in `src/data.py`:

- Load the three CSVs.
- Create `next_return` and `target`.
- Remove leakage columns.
- Engineer trailing return/volatility and price-vs-EMA features.
- Return `X_train, X_test, y_train, y_test` for the project template.

### Completed

Implemented `src/data.py`.

Smoke-test results:

- Clean modeling dataset: 10,407 rows, 96 total columns.
- Model feature matrix: 89 features.
- Train plus validation rows: 9,063.
- Test rows: 1,344.
- Test period starts in 2022.
- No missing feature values after preprocessing.
- Leakage columns are not present in the final feature list.

Next implementation task: create `scripts/train_models.py` and train the first baseline models.

### Feature/Processed Data Update

Added `scripts/prepare_data.py` and saved the processed datasets:

- `data/processed_modeling_dataset.csv`
- `data/processed_train.csv`
- `data/processed_validation.csv`
- `data/processed_test.csv`

Added additional non-leaky finance features:

- Volatility-adjusted return features.
- Short-vs-long volatility ratio.
- Rolling drawdown from 20-day and 60-day highs.
- Distance from 20-day low.
- EMA trend spread features.

These are useful because they capture momentum, volatility regime, trend, and drawdown state without using future information.

### Model Training Update

Added `scripts/train_models.py` and trained three supervised classifiers:

- Logistic Regression
- Random Forest
- Hist Gradient Boosting

Saved models:

- `models/log_reg.joblib`
- `models/random_forest.joblib`
- `models/hist_gradient_boosting.joblib`

Saved report:

- `results/training_classification_report.csv`

Initial untouched test results:

| Model | Accuracy | Balanced accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Majority-up baseline | 0.4903 | 0.5000 | 0.6580 | n/a |
| Previous-day direction baseline | 0.5179 | 0.5177 | 0.5083 | n/a |
| Logistic Regression | 0.7351 | 0.7351 | 0.7315 | 0.8090 |
| Random Forest | 0.6778 | 0.6767 | 0.6528 | 0.7226 |
| Hist Gradient Boosting | 0.7106 | 0.7105 | 0.7060 | 0.7691 |

Current best classifier: Logistic Regression.

Registered model test metrics from `results/model_metrics.csv`:

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7359 | 0.7353 | 0.7420 | 0.7071 | 0.7242 |
| Random Forest | 0.6763 | 0.6754 | 0.6867 | 0.6252 | 0.6545 |
| Hist Gradient Boosting | 0.6897 | 0.6909 | 0.6618 | 0.7511 | 0.7036 |

Next task: build strategy/backtest evaluation using predicted probabilities and long/cash thresholds.

### Backtest Update

Added `src/backtest.py` and `scripts/run_backtest.py`.

The strategy is an equal-weight daily portfolio across the three index signals. For model strategies:

1. Train a temporary model on the train split only.
2. Tune the long/cash probability threshold on validation only.
3. Apply the selected threshold to final saved models on the untouched test split.
4. Include a 1 basis point cost per signal change.

Saved outputs:

- `results/strategy_metrics.csv`
- `results/equity_curves.csv`
- `results/strategy_returns.csv`

Initial test-period strategy results:

| Strategy | Threshold | Total return | Annualized return | Sharpe | Max drawdown | Exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Buy and hold | n/a | -0.1014 | -0.0411 | -0.2102 | -0.2549 | 1.0000 |
| Previous-day direction | n/a | -0.0729 | -0.0346 | -0.2748 | -0.1470 | 0.4903 |
| Positive 20d momentum | n/a | -0.0534 | -0.0255 | -0.2439 | -0.1450 | 0.4784 |
| Logistic Regression long/cash | 0.55 | 2.9924 | 0.7862 | 6.9709 | -0.0215 | 0.4122 |
| Random Forest long/cash | 0.60 | 1.9585 | 0.6167 | 5.6873 | -0.0224 | 0.3482 |
| Hist Gradient Boosting long/cash | 0.50 | 2.5032 | 0.7137 | 5.7918 | -0.0234 | 0.5565 |

These results are strong enough that the next review step should include additional stress tests:

- Feature ablation without `price_vs_ema_*`.
- Per-index strategy metrics.
- Higher transaction cost sensitivity.
- Walk-forward validation instead of a single train/validation/test split.
