# filename: src/evaluate.py
import os
import sys
import subprocess

# STAGE 1: Install dependencies BEFORE any other imports
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

# STAGE 2: Standard Imports
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
    # Fix for unpickling custom classes
    import __main__
    __main__.VotingModel = VotingModel

    s3 = boto3.client("s3")
    bucket = "sg-home-credit"

    # Define paths
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
    # 1) Get feature names and categorical features from model
    # ---------------------------------------------------------
    base_est = model.estimators[0]
    trained_cols = list(base_est.feature_name_)
    cat_feats = list(getattr(base_est, "categorical_feature_", []))

    # ---------------------------------------------------------
    # 2) Align columns with training (order + subset)
    #    - keep only columns the model knows
    #    - reindex to exact training order
    # ---------------------------------------------------------
    X_test = X_test.reindex(columns=trained_cols)

    # ---------------------------------------------------------
    # 3) Ensure categorical columns have dtype 'category'
    # ---------------------------------------------------------
    for col in cat_feats:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    print("Predicting...")
    y_pred = model.predict_proba(X_test)[:, 1]

    # Save results
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
