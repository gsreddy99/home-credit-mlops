# filename: src/evaluate.py
import os
import sys
import subprocess
import boto3
import pandas as pd
import joblib
import tempfile
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# 1. BOOTSTRAP: Install dependencies from the pipeline-injected requirements file
def install_requirements():
    req_path = "/opt/ml/processing/input/reqs/requirements.txt"
    if os.path.exists(req_path):
        print(f"Installing dependencies from {req_path}...")
        try:
            # Upgrade pip for better resolution and install pinned requirements
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", req_path])
            import importlib
            importlib.invalidate_caches()
        except Exception as e:
            print(f"Failed to install requirements: {e}")
            sys.exit(1)

install_requirements()

# 2. MONKEY PATCH: Fix NumPy 2.0 compatibility for legacy sub-dependencies
def patch_numpy():
    if not hasattr(np, "bool"):
        np.bool = bool
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int
    print("Applied NumPy 2.0 compatibility patches (bool, float, int)")

patch_numpy()

BUCKET = "sg-home-credit"

# 3. UPDATED VOTING MODEL (Matches Training)
class VotingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

def evaluate_and_update(output_dir: str):
    # Register the class in the main module for the unpickler
    import __main__
    __main__.VotingModel = VotingModel

    s3 = boto3.client("s3")
    model_key = "home-credit/model/aiml_model.pkl"
    test_key = "home-credit/silver/test/test.csv"
    sample_key = "home-credit/model/sample_suggestions.csv"

    def download_s3(key, required=True):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                s3.download_file(BUCKET, key, tmp.name)
                return tmp.name
            except Exception as e:
                if not required: return None
                raise e

    print("Downloading assets from S3...")
    model_path = download_s3(model_key)
    test_path = download_s3(test_key)
    sample_path = download_s3(sample_key, required=False)

    print("Loading model and performing inference...")
    model = joblib.load(model_path)
    df_test = pd.read_csv(test_path)

    # Drop non-feature columns
    X_test = df_test.drop(columns=["case_id", "WEEK_NUM", "target"], errors="ignore")

    print(f"Inference for {len(X_test)} records...")
    # Use index 1 for positive class probability
    y_pred = model.predict_proba(X_test)[:, 1]

    df_result = pd.DataFrame({"case_id": df_test["case_id"], "score": y_pred})

    if sample_path:
        df_sample = pd.read_csv(sample_path)
        df_result = df_sample[["case_id"]].merge(df_result, on="case_id", how="left")
        df_result["score"] = df_result["score"].fillna(0.005)

    os.makedirs(output_dir, exist_ok=True)
    local_path = os.path.join(output_dir, "sample_suggestions.csv")
    df_result.to_csv(local_path, index=False)

    s3.upload_file(local_path, BUCKET, "home-credit/gold/sample_suggestions.csv")
    print("Process complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/evaluation")
    args = parser.parse_args()
    evaluate_and_update(args.output_dir)