# filename: src/evaluate.py
import os
import sys
import subprocess

# 1. BOOTSTRAP: Install modern dependencies (including Polars)
def install_deps():
    req_path = "/opt/ml/processing/input/reqs/requirements.txt"
    if os.path.exists(req_path):
        print("Installing modern ML stack (NumPy 2.0, Polars, LightGBM)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", req_path])

        # REINFORCED BRIDGE: Maps both the base and the multiarray submodule
        import numpy
        import numpy.core.multiarray as multiarray
        sys.modules["numpy._core"] = numpy
        sys.modules["numpy._core.multiarray"] = multiarray
        print("✓ NumPy 2.0 namespaces bridged for model compatibility.")

install_deps()

# 2. IMPORTS
import boto3
import pandas as pd
import polars as pl
import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

BUCKET = "sg-home-credit"

# VotingModel class definition (must match training)
class VotingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators

    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

def evaluate_and_update():
    # Ensure VotingModel is in the main namespace for joblib
    import __main__
    __main__.VotingModel = VotingModel

    s3 = boto3.client("s3")
    output_dir = "/opt/ml/processing/evaluation"

    # S3 Paths
    model_key = "home-credit/model/aiml_model.pkl"
    test_key = "home-credit/silver/test/test.csv"

    # Download assets
    test_path, model_path = "/tmp/test.csv", "/tmp/model.pkl"
    print("Downloading assets from S3...")
    s3.download_file(BUCKET, test_key, test_path)
    s3.download_file(BUCKET, model_key, model_path)

    # Use POLARS for fast data loading
    print("Loading test data with Polars...")
    df = pl.read_csv(test_path)

    # Drop non-feature columns and convert to pandas for the model
    X_test = df.drop(["case_id", "WEEK_NUM", "target"], strict=False).to_pandas()

    print("Unpickling model...")
    model = joblib.load(model_path)

    print("Generating predictions...")
    y_pred = model.predict_proba(X_test)[:, 1]

    # Create result with Polars
    result = pl.DataFrame({
        "case_id": df["case_id"],
        "score": y_pred
    })

    os.makedirs(output_dir, exist_ok=True)
    local_csv = os.path.join(output_dir, "sample_suggestions.csv")
    result.write_csv(local_csv)

    s3.upload_file(local_csv, BUCKET, "home-credit/gold/sample_suggestions.csv")
    print("Inference successful and uploaded to Gold layer.")

if __name__ == "__main__":
    evaluate_and_update()