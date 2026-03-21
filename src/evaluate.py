# filename: src/evaluate.py
import os
import sys
import subprocess

def install_and_bridge():
    req_path = "/opt/ml/processing/input/reqs/requirements.txt"
    if os.path.exists(req_path):
        print("Installing Polars, LightGBM, and NumPy 2.0...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", req_path])

        # THE FIX: This bridges the gap for models saved with NumPy 2.x
        import numpy
        import numpy.core.multiarray as multiarray
        sys.modules["numpy._core"] = numpy
        sys.modules["numpy._core.multiarray"] = multiarray
        print("✓ NumPy namespaces bridged.")

install_and_bridge()

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
        y_preds = [est.predict_proba(X) for est in self.estimators]
        return np.mean(y_preds, axis=0)

def main():
    import __main__
    __main__.VotingModel = VotingModel

    s3 = boto3.client("s3")
    bucket = "sg-home-credit" # Or parse from args

    # Download assets
    test_path, model_path = "/tmp/test.csv", "/tmp/model.pkl"
    s3.download_file(bucket, "home-credit/silver/test/test.csv", test_path)
    s3.download_file(bucket, "home-credit/model/aiml_model.pkl", model_path)

    # Use POLARS for speed
    df = pl.read_csv(test_path)
    X_test = df.drop(["case_id", "WEEK_NUM", "target"], strict=False).to_pandas()

    print("Loading model...")
    model = joblib.load(model_path)
    y_pred = model.predict_proba(X_test)[:, 1]

    # Output results
    result = pl.DataFrame({"case_id": df["case_id"], "score": y_pred})
    output_file = "/opt/ml/processing/evaluation/sample_suggestions.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result.write_csv(output_file)

    s3.upload_file(output_file, bucket, "home-credit/gold/sample_suggestions.csv")
    print("Inference Complete.")

if __name__ == "__main__":
    main()