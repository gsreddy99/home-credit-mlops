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

    cols_to_drop = [c for c in ["case_id", "WEEK_NUM", "target"] if c in df.columns]
    X_test = df.drop(cols_to_drop).to_pandas()

    print("Loading model...")
    model = joblib.load(model_path)
    base_est = model.estimators[0]

    trained_cols = list(base_est.feature_name_)
    trained_cats = list(getattr(base_est, "categorical_feature_", []))

    print("\n================ TRAINED FEATURE COUNT ================")
    print(len(trained_cols))

    print("\n================ TEST FEATURE COUNT ================")
    print(len(X_test.columns))

    missing = [c for c in trained_cols if c not in X_test.columns]
    extra = [c for c in X_test.columns if c not in trained_cols]

    print("\n================ MISSING FEATURES ================")
    print(missing)

    print("\n================ EXTRA FEATURES ================")
    print(extra)

    # Align columns (missing become NaN)
#     X_test = X_test.reindex(columns=trained_cols)
#
#     # ---------------------------------------------------------
#     # BYPASS ALL LIGHTGBM FEATURE VALIDATION
#     # ---------------------------------------------------------
#     for est in model.estimators:
#         booster = est._Booster
#
#         # LightGBM expects lists, not booleans
#         booster.pandas_categorical = []
#         booster.categorical_feature = []
#
#         # Force feature_name to match the input
#         booster.feature_name = trained_cols
#
#     print("\nPredicting...")
#     y_pred = model.predict_proba(X_test)[:, 1]
#
#     output_df = pl.DataFrame({
#         "case_id": df["case_id"],
#         "score": y_pred
#     })
#
#     local_csv = os.path.join(output_dir, "sample_suggestions.csv")
#     output_df.write_csv(local_csv)
#
#     print("Uploading results to Gold layer...")
#     s3.upload_file(local_csv, bucket, "home-credit/gold/sample_suggestions.csv")
#     print("Execution complete.")


if __name__ == "__main__":
    main()
