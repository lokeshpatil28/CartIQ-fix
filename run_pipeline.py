"""
Runs the full data -> training pipeline end to end, in order.
Usage: python run_pipeline.py
(Run backend/api.py via uvicorn and frontend/frontend_app.py via streamlit
separately - see README.md.)
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("data_generation", "generate_items.py"),
    ("data_generation", "generate_orders.py"),
    ("data_generation", "build_snapshots.py"),
    ("data_generation", "generate_training_data.py"),
    ("data_generation", "phase2_feature_engineering.py"),
    ("model_training", "train_ranker.py"),
]


def main():
    for folder, script in STEPS:
        cwd = ROOT / folder
        print(f"\n{'=' * 70}\nRunning {folder}/{script}\n{'=' * 70}")
        result = subprocess.run([sys.executable, script], cwd=cwd)
        if result.returncode != 0:
            print(f"\nPipeline stopped: {folder}/{script} exited with code {result.returncode}")
            sys.exit(result.returncode)

    print("\nPipeline complete. Model saved to model_training/model.joblib")
    print("Next: uvicorn backend.api:app --reload   (from the project root)")
    print("Then: streamlit run frontend/frontend_app.py")


if __name__ == "__main__":
    main()
