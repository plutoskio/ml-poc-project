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
- HistGradientBoosting: main performance candidate.

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

Current smoke-test results after the senior review fix:

- Clean modeling dataset: 10,227 rows, 100 total columns.
- Model feature matrix: 93 features.
- Train plus validation rows: 8,883.
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

These are useful because they capture momentum, volatility regime, trend, and drawdown state without using future information. A later review found that the dataset-provided `EMA_*` columns were actually future-looking, so the corrected pipeline now excludes them and recomputes trailing EMAs from past prices only.

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

The senior review later invalidated the first training results because they were driven by future-looking EMA features.

Corrected registered model test metrics from `results/model_metrics.csv`:

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.4978 | 0.5010 | 0.4911 | 0.6692 | 0.5665 |
| Random Forest | 0.5186 | 0.5218 | 0.5067 | 0.6859 | 0.5828 |
| Hist Gradient Boosting | 0.5089 | 0.5145 | 0.4995 | 0.8027 | 0.6158 |

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

The first high-return backtest was invalidated by the senior review because it depended on future-looking EMA features.

### Senior Review And Leakage Fix

The senior review found two disqualifying issues:

1. The raw dataset `EMA_10`, `EMA_20`, `EMA_50`, and `EMA_200` columns were future-looking. They follow a backward EMA recurrence and were not available at time `t`.
2. Same-day close-derived features were being used to enter a close-to-close strategy, which creates execution timing bias.

Fixes implemented:

- Added raw `EMA_*` columns to the leakage exclusion list.
- Recomputed trailing EMAs from `Price` using only past prices.
- Renamed valid EMA features to `price_vs_trailing_ema_*`.
- Shifted all model feature columns by one trading day within each index.
- Removed misleading validation rows from the final-model backtest output because final models are trained on train plus validation.
- Regenerated processed data, models, model metrics, and backtest outputs.

Corrected processed dataset summary:

| Dataset | Rows | Columns |
| --- | ---: | ---: |
| Full processed modeling dataset | 10,227 | 100 |
| Train | 7,368 | 100 |
| Validation | 1,515 | 100 |
| Test | 1,344 | 100 |

Corrected test-period strategy results:

| Strategy | Threshold | Total return | Annualized return | Sharpe | Max drawdown | Exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Buy and hold | n/a | -0.1014 | -0.0411 | -0.2102 | -0.2549 | 1.0000 |
| Previous-day direction | n/a | -0.2518 | -0.1552 | -1.2371 | -0.2697 | 0.4896 |
| Price above trailing EMA 20 | n/a | -0.1097 | -0.0596 | -0.5558 | -0.1692 | 0.4926 |
| Positive 20d momentum | n/a | -0.1226 | -0.0670 | -0.5858 | -0.1874 | 0.4807 |
| Logistic Regression long/cash | 0.60 | 0.0966 | 0.0572 | 0.5531 | -0.0902 | 0.2552 |
| Random Forest long/cash | 0.60 | -0.0006 | 0.0020 | 0.0294 | -0.0645 | 0.0915 |
| Hist Gradient Boosting long/cash | 0.60 | -0.0640 | -0.0293 | -0.2326 | -0.1486 | 0.3854 |

Corrected interpretation:

- The original high Sharpe result was not valid.
- After removing leakage and lagging features, prediction is close to chance.
- Logistic Regression shows a modest positive test-period long/cash result, but it is not strong enough to claim a robust trading edge without more validation.

Remaining stress tests:

- Per-index strategy metrics.
- Higher transaction cost sensitivity.
- Walk-forward validation instead of a single train/validation/test split.
- Probability calibration check.

### Repository Update

Earlier commit pushed to GitHub before the leakage correction:

```text
commit c241856 - Build supervised quant finance pipeline
origin/main
```

Current corrected work is local and should be committed after review. Remaining work:

- Streamlit app customization.
- Robustness/stress tests for the corrected modest edge.
- Final narrative and charts for the project presentation.

### Plot Generation

Added `scripts/generate_plots.py` and generated corrected post-leakage visual outputs:

- `plots/test_equity_curves.png`
- `plots/test_drawdowns.png`
- `plots/test_strategy_metric_bars.png`
- `plots/test_model_metric_bars.png`
- `plots/test_return_distribution.png`
- `plots/test_model_exposure.png`

These plots use the corrected results after excluding future-looking EMA columns and lagging all model features by one trading day.

### Engineering Cleanup

Addressed the senior engineering review items:

- Added project packaging/configuration in `pyproject.toml` so modules can import from `src` without `sys.path` hacks.
- Added `setup.cfg` and `requirements-dev.txt` for linting, formatting, and tests.
- Moved shared model definitions/probability helpers into `src/modeling.py`.
- Reused `src.metrics.compute_classification_metrics()` from training instead of duplicating metric logic.
- Added focused tests for data leakage rules, target creation, lagged features, classification metrics, and backtest mechanics.
- Ran formatting and quality checks successfully:
  - `python -m pytest`
  - `python -m compileall src scripts tests`
  - `python -m black --check src scripts tests`
  - `python -m flake8 src scripts tests`
  - `python -m pylint src scripts tests`
