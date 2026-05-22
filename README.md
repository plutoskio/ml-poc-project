# Walmart Sales Forecasting ML Proof of Concept

This project is a supervised machine learning proof of concept for retail
demand forecasting. It predicts daily item-level unit sales for Walmart stores
using historical sales, calendar features, store/item identifiers, and local
weather observations.

The business goal is inventory planning: better forecasts can help store teams
anticipate demand, reduce stockouts, avoid excess inventory, and identify the
store-product combinations that need attention.

## Project Summary

- Problem type: supervised regression
- Target: `units`, the number of units sold for one store, item, and date
- Dataset: Walmart Recruiting II: Sales in Stormy Weather
- Source: Kaggle competition dataset
- Evaluation split: chronological train/test split
- Training period: `2012-01-01` to `2014-06-30`
- Test period: `2014-07-01` to `2014-10-31`

The original Kaggle test file does not include labels, so this project creates
its own chronological test split from the labeled `train.csv` file. This is more
realistic than a random split because a forecasting model should train on past
dates and evaluate on future dates.

## Models

The project compares three supervised models:

- Lag Blend Baseline: transparent benchmark using lag and rolling sales
  features.
- Weighted Poisson Regression: count-regression model with categorical
  encoding, numeric scaling, and sample weights for the zero-heavy target.
- Histogram Gradient Boosting: nonlinear tabular model trained on engineered
  sales, calendar, and weather features.

Current best model: Histogram Gradient Boosting.

Current chronological test metrics:

| Model | MAE | RMSE | RMSLE | R2 | Positive-sales MAE | Zero-sales MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Histogram Gradient Boosting | 0.222 | 2.404 | 0.093 | 0.893 | 9.261 | 0.009 |
| Weighted Poisson Regression | 1.404 | 6.598 | 0.623 | 0.197 | 29.725 | 0.738 |
| Lag Blend Baseline | 1.238 | 2.960 | 0.694 | 0.838 | 11.086 | 1.006 |

## Repository Structure

```text
deliverables/       Assignment markdown files
data/               Local raw and processed data, ignored by Git
models/             Saved trained models
notebooks/          Exploratory data analysis notebook
plots/              Saved static plots
results/            Metrics, summaries, predictions, feature importance
scripts/            Pipeline scripts
src/                Project source code
tests/              Unit tests
```

## Assignment Deliverables

- `deliverables/assignment1.md`: project topic and dataset description.
- `deliverables/assignment2.md`: feature engineering and preprocessed dataset.
- `deliverables/assignment3.md`: three model descriptions, saved models, and
  model comparison.
- `plots/`: required static plots for EDA, model comparison, and best-model
  results.
- `src/app.py`: completed Streamlit dashboard.
- This README: project description and data download guide.

## How To Get The Data

The data is not supposed to be stored in Git. Download it locally from Kaggle:

```text
https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather/data
```

You need a Kaggle account and may need to accept the competition rules before
the files are available.

### Option 1: Download From The Kaggle Website

1. Open the Kaggle data page:

   ```text
   https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather/data
   ```

2. Download the dataset archive.

3. Create this local folder inside the repo:

   ```bash
   mkdir -p data/raw/walmart-recruiting-sales-in-stormy-weather
   ```

4. Unzip the downloaded files into:

   ```text
   data/raw/walmart-recruiting-sales-in-stormy-weather/
   ```

5. Confirm the folder contains at least:

   ```text
   train.csv
   key.csv
   weather.csv
   sampleSubmission.csv
   ```

### Option 2: Download With The Kaggle CLI

Install and configure the Kaggle CLI first:

```bash
pip install kaggle
```

Then download the competition files:

```bash
mkdir -p data/raw/walmart-recruiting-sales-in-stormy-weather
kaggle competitions download \
  -c walmart-recruiting-sales-in-stormy-weather \
  -p data/raw/walmart-recruiting-sales-in-stormy-weather
```

Unzip the downloaded archive:

```bash
unzip data/raw/walmart-recruiting-sales-in-stormy-weather/*.zip \
  -d data/raw/walmart-recruiting-sales-in-stormy-weather
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For tests:

```bash
pip install -r requirements-dev.txt
```

## Rebuild The Project

After downloading the raw data, generate the processed dataset and EDA outputs:

```bash
python scripts/prepare_data.py
```

Train and save the models:

```bash
python scripts/train_models.py
```

Generate static plots:

```bash
python scripts/generate_plots.py
```

Run the template entry point, evaluate saved models, and launch Streamlit:

```bash
python scripts/main.py
```

The Streamlit dashboard opens at:

```text
http://localhost:8501
```

## Quality Checks

Run the unit tests:

```bash
python -m pytest
```

The tests cover weather cleaning, leakage-safe lag feature creation, and metric
computation.

## Main Files

- `src/data.py`: loads the processed dataset and returns `X_train`, `X_test`,
  `y_train`, `y_test`.
- `src/features.py`: cleans weather data, merges tables, and builds lag/rolling
  features.
- `src/modeling.py`: defines the baseline, Poisson regression, and histogram
  gradient boosting models.
- `src/metrics.py`: computes regression metrics.
- `src/app.py`: Streamlit dashboard.
- `PROJECT_APPROACH.md`: concise explanation of the full project methodology.
- `deliverables/assignment1.md`: dataset and project-topic deliverable.
- `deliverables/assignment2.md`: feature engineering and preprocessed dataset
  deliverable.
- `deliverables/assignment3.md`: model description and comparison deliverable.
