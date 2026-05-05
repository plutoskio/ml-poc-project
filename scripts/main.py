from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(ROOT_DIR / ".env")

from config import MODELS, RESULTS_DIR, STREAMLIT_HOST, STREAMLIT_PORT  # noqa: E402
from data import load_dataset_split  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from results import save_best_model_predictions  # noqa: E402


def evaluate_models() -> pd.DataFrame:
    """Evaluate registered models and save a comparison table."""
    if not MODELS:
        raise RuntimeError("No models are registered in src/config.py.")

    _, X_test, _, y_test = load_dataset_split()
    rows = []

    for model_id, model_config in MODELS.items():
        model_path = Path(model_config["path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file for {model_id}: {model_path}")

        model = joblib.load(model_path)
        if not hasattr(model, "predict"):
            raise TypeError(f"Model {model_id} does not expose a predict method.")

        y_pred = model.predict(X_test)
        metric_values = compute_metrics(y_test, y_pred)
        rows.append(
            {
                "model_id": model_id,
                "model_name": model_config.get("name", model_id),
                **metric_values,
            }
        )

    results = pd.DataFrame(rows).sort_values("rmsle")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    save_best_model_predictions(results, X_test, y_test)
    return results


def launch_streamlit() -> None:
    """Launch the Streamlit app using the fixed src/app.py entry point."""
    app_path = ROOT_DIR / "src" / "app.py"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address",
            STREAMLIT_HOST,
            "--server.port",
            str(STREAMLIT_PORT),
        ],
        check=True,
    )


def main() -> None:
    metrics = evaluate_models()
    print(metrics.to_string(index=False))
    launch_streamlit()


if __name__ == "__main__":
    main()
