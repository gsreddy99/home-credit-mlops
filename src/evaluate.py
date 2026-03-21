# filename: src/evaluate.py
import os
import sys
import subprocess

def prepare_environment():
    req_path = "/opt/ml/processing/input/reqs/requirements.txt"
    if os.path.exists(req_path):
        print("Upgrading core dependencies (NumPy 2.0 + PyArrow)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--no-cache-dir",
            "-r", req_path
        ])
        print("✓ Environment ready.")

prepare_environment()

import boto3
import pandas as pd
import polars as pl
import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class VotingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators

    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

def main():
    import __main__
    __main__.VotingModel = VotingModel

    s3 = boto3.client("s3")
    bucket = "sg-home-credit"

    test_path, model_path = "/tmp/test.csv", "/tmp/model.pkl"
    output_dir = "/opt/ml/processing/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading assets...")
    s3.download_file(bucket, "home-credit/silver/test/test.csv", test_path)
    s3.download_file(bucket, "home-credit/model/aiml_model.pkl", model_path)

    print("Processing with Polars...")
    df = pl.read_csv(test_path)

    # Drop non-feature columns
    cols_to_drop = [c for c in ["case_id", "WEEK_NUM", "target"] if c in df.columns]
    X_test = df.drop(cols_to_drop).to_pandas()

    print("Loading model...")
    model = joblib.load(model_path)

    # ---------------------------------------------------------
    # FIX 1: Restore categorical dtype (required by LightGBM)
    # ---------------------------------------------------------
    for col in X_test.select_dtypes(include=["object"]).columns:
        X_test[col] = X_test[col].astype("category")

    # ---------------------------------------------------------
    # FIX 2: Align columns with model training order
    # ---------------------------------------------------------
    trained_cols = model.estimators[0].feature_name_
    X_test = X_test.reindex(columns=trained_cols)

    print("Predicting...")
    y_pred = model.predict_proba(X_test)[:, 1]

    output_df = pl.DataFrame({
        "case_id": df["case_id"],
        "score": y_pred
    })

    local_csv = os.path.join(output_dir, "sample_suggestions.csv")
    output_df.write_csv(local_csv)

    print("Uploading results to Gold layer...")
    s3.upload_file(local_csv, bucket, "home-credit/gold/sample_suggestions.csv")
    print("Execution complete.")

if __name__ == "__main__":
    main()
