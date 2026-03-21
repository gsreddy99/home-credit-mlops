# filename: src/evaluate.py
import os
import sys
import subprocess

# --- STAGE 1: INSTALL DEPENDENCIES ---
def install_deps():
    req_path = "/opt/ml/processing/input/reqs/requirements.txt"
    if os.path.exists(req_path):
        # We install dependencies but NO LONGER bridge the namespaces.
        # NumPy 2.0 is now natively present in the environment.
        print("Installing Polars, LightGBM, and NumPy 2.0...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", req_path])

install_deps()

# --- STAGE 2: IMPORTS ---
# Imports must happen AFTER pip install is complete
import boto3
import pandas as pd
import polars as pl
import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

BUCKET = "sg-home-credit"

class VotingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators
    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

def main():
    # Ensure the class is available for unpickling
    import __main__
    __main__.VotingModel = VotingModel

    s3 = boto3.client("s3")
    output_dir = "/opt/ml/processing/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    test_path, model_path = "/tmp/test.csv", "/tmp/model.pkl"
    print("Downloading assets from S3...")
    s3.download_file(BUCKET, "home-credit/silver/test/test.csv", test_path)
    s3.download_file(BUCKET, "home-credit/model/aiml_model.pkl", model_path)

    print("Loading data with Polars...")
    df = pl.read_csv(test_path)

    # Feature selection
    cols_to_drop = [c for c in ["case_id", "WEEK_NUM", "target"] if c in df.columns]
    X_test = df.drop(cols_to_drop).to_pandas()

    print("Unpickling model (Native NumPy 2.0)...")
    model = joblib.load(model_path)

    print("Generating scores...")
    y_pred = model.predict_proba(X_test)[:, 1]

    output_df = pl.DataFrame({
        "case_id": df["case_id"],
        "score": y_pred
    })

    local_csv = os.path.join(output_dir, "sample_suggestions.csv")
    output_df.write_csv(local_csv)

    print("Uploading results...")
    s3.upload_file(local_csv, BUCKET, "home-credit/gold/sample_suggestions.csv")
    print("Done!")

if __name__ == "__main__":
    main()