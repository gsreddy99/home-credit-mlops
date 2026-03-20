# filename: src/evaluate.py

import subprocess
subprocess.run(["pip", "install", "--upgrade", "numpy"], check=True)
subprocess.run(["pip", "install", "lightgbm"], check=True)

import os
import argparse
import boto3
import pandas as pd
import joblib
import tempfile

BUCKET = "sg-home-credit"

# ----------------------------------------------------------------------
# Add the VotingModel class EXACTLY as defined in train.py
# ----------------------------------------------------------------------
class VotingModel:
    def __init__(self, estimators):
        self.estimators = estimators

    def predict_proba(self, X):
        probs = [est.predict_proba(X) for est in self.estimators]
        return sum(probs) / len(probs)


# ----------------------------------------------------------------------
# Helper to download files from S3
# ----------------------------------------------------------------------
def download_file_from_s3(key):
    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        s3.download_file(BUCKET, key, tmp.name)
        return tmp.name


# ----------------------------------------------------------------------
# Evaluation logic
# ----------------------------------------------------------------------
def evaluate_and_update(output_dir: str):

    s3 = boto3.client("s3")

    # Updated model + test + template paths
    model_key   = "home-credit/model/aiml_model.pkl"
    test_key    = "home-credit/silver/test/test.csv"
    sample_key  = "home-credit/model/sample_submission.csv"   # corrected filename

    print(f"Loading model:     s3://{BUCKET}/{model_key}")
    print(f"Loading test:      s3://{BUCKET}/{test_key}")
    print(f"Loading template:  s3://{BUCKET}/{sample_key}")

    model_path  = download_file_from_s3(model_key)
    test_path   = download_file_from_s3(test_key)
    sample_path = download_file_from_s3(sample_key)

    # Load model (now works because VotingModel is defined above)
    model = joblib.load(model_path)

    df_test = pd.read_csv(test_path)
    df_sample = pd.read_csv(sample_path)

    # Predict
    X_test = df_test.drop(columns=["case_id", "WEEK_NUM"], errors="ignore")
    X_test.index = df_test["case_id"]

    print("Generating predictions...")
    y_pred = model.predict_proba(X_test)[:, 1]

    df_pred = pd.DataFrame({"case_id": X_test.index, "score": y_pred})

    df_result = df_sample[["case_id"]].merge(df_pred, on="case_id", how="left")
    df_result["score"] = df_result["score"].fillna(0.005)

    # Save & upload
    os.makedirs(output_dir, exist_ok=True)
    local_path = os.path.join(output_dir, "sample_submission.csv")
    df_result.to_csv(local_path, index=False)

    gold_key = "home-credit/gold/sample_submission.csv"
    s3.upload_file(local_path, BUCKET, gold_key)

    print(f"✓ Updated predictions uploaded to: s3://{BUCKET}/{gold_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/evaluation")
    args = parser.parse_args()

    evaluate_and_update(args.output_dir)
