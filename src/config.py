from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

for directory in [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    directory.mkdir(exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

MODELS = {
    "log_reg": {
        "name": "Logistic Regression",
        "description": "Linear supervised baseline with standardized features.",
        "path": MODELS_DIR / "log_reg.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Tree ensemble benchmark for nonlinear market interactions.",
        "path": MODELS_DIR / "random_forest.joblib",
    },
    "hist_gradient_boosting": {
        "name": "Hist Gradient Boosting",
        "description": "Gradient boosting classifier for tabular market features.",
        "path": MODELS_DIR / "hist_gradient_boosting.joblib",
    },
}
