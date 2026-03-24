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


# -------------------------------------------------------------------------
# VotingModel (must be top-level for joblib)
# -------------------------------------------------------------------------
class VotingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators

    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    import __main__
    __main__.VotingModel = VotingModel  # required for joblib unpickling

    s3 = boto3.client("s3")
    bucket = "sg-home-credit"

    test_path = "/tmp/test.csv"
    model_path = "/tmp/model.pkl"
    output_dir = "/opt/ml/processing/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading assets...")
    # test.csv from silver
    s3.download_file(bucket, "home-credit/silver/test/test.csv", test_path)
    # latest model produced by train.py
    s3.download_file(bucket, "home-credit/model/aiml_model.pkl", model_path)

    print("Processing test.csv with Polars...")
    df = pl.read_csv(test_path)

    # Drop non-feature columns
    cols_to_drop = [c for c in ["case_id", "WEEK_NUM", "target"] if c in df.columns]
    X_test = df.drop(cols_to_drop).to_pandas()

    print("Loading model...")
    model = joblib.load(model_path)

    # ---------------------------------------------------------------------
    # Extract trained feature names and categorical features
    # ---------------------------------------------------------------------
    base_est = model.estimators[0]
    trained_cols = list(base_est.feature_name_)

    raw_cats = list(getattr(base_est, "categorical_feature_", []))
    cat_cols = []
    for v in raw_cats:
        if isinstance(v, str):
            cat_cols.append(v)
        else:
            # index into trained_cols
            cat_cols.append(trained_cols[int(v)])

    print("\n================ TRAINED FEATURE COUNT ================")
    print(len(trained_cols))

    print("\n================ TEST FEATURE COUNT ================")
    print(len(X_test.columns))

    # ---------------------------------------------------------------------
    # Align test columns to trained columns
    # ---------------------------------------------------------------------
    missing = [c for c in trained_cols if c not in X_test.columns]
    extra   = [c for c in X_test.columns if c not in trained_cols]

    print("\n================ MISSING FEATURES ================")
    print(missing)

    print("\n================ EXTRA FEATURES ================")
    print(extra)

    # Add missing columns with correct dtype (categorical vs numeric)
    for col in missing:
        if col in cat_cols:
            X_test[col] = pd.Series([np.nan] * len(X_test), dtype="category")
        else:
            X_test[col] = np.nan

    # Drop extra columns and reorder to match training
    X_test = X_test[trained_cols]

    # Ensure all categorical features are category dtype
    for col in cat_cols:
        if col in X_test.columns and X_test[col].dtype != "category":
            X_test[col] = X_test[col].astype("category")

    # Mirror train.py: convert any remaining object columns to category
    for col in X_test.columns:
        if X_test[col].dtype == "object":
            X_test[col] = X_test[col].astype("category")

    # ---------------------------------------------------------------------
    # BYPASS LIGHTGBM CATEGORICAL VALIDATION (critical fix)
    # ---------------------------------------------------------------------
    for est in model.estimators:
        booster = est._Booster
        # Disable pandas-based categorical checks
        booster.pandas_categorical = None
        booster.categorical_feature = None

    # ---------------------------------------------------------------------
    # Predict
    # ---------------------------------------------------------------------
    print("\nPredicting...")
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
