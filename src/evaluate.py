# filename: src/evaluate.py

import os
import argparse
import sys
import subprocess
import boto3
import pandas as pd
import joblib
import tempfile


BUCKET = "sg-home-credit"


def install_lightgbm_if_missing():
    """Install lightgbm at runtime if not already present"""
    try:
        import lightgbm
        print("lightgbm is already installed")
    except ImportError:
        print("lightgbm not found → installing now...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--no-cache-dir", "lightgbm>=4.0.0"
            ])
            print("lightgbm installed successfully")
            import lightgbm  # verify
        except Exception as e:
            print(f"Failed to install lightgbm: {e}", file=sys.stderr)
            raise


def download_file_from_s3(key, required=True, description="file"):
    """Download from S3 with better error handling"""
    s3 = boto3.client("s3")
    full_path = f"s3://{BUCKET}/{key}"
    print(f"Downloading {description}: {full_path}")

    try:
        # Check if object exists
        s3.head_object(Bucket=BUCKET, Key=key)
    except s3.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        if error_code == '404':
            msg = f"File not found: {full_path}"
            if required:
                print(msg, file=sys.stderr)
                raise FileNotFoundError(msg)
            else:
                print(msg + " → continuing with fallback")
                return None
        else:
            print(f"S3 error: {e}", file=sys.stderr)
            raise

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            s3.download_file(BUCKET, key, tmp.name)
            print(f"Downloaded successfully: {full_path}")
            return tmp.name
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        raise


class VotingModel:
    def __init__(self, estimators):
        self.estimators = estimators

    def predict_proba(self, X):
        probs = [est.predict_proba(X) for est in self.estimators]
        return sum(probs) / len(probs)


def evaluate_and_update(output_dir: str):
    # Install lightgbm before trying to load the model
    install_lightgbm_if_missing()

    s3 = boto3.client("s3")

    # ── Input paths ──────────────────────────────────────────────────────────────
    model_key   = "home-credit/model/aiml_model.pkl"
    test_key    = "home-credit/silver/test/test.csv"
    sample_key  = "home-credit/model/sample_suggestions.csv"   # change if needed

    print(f"Loading model:    s3://{BUCKET}/{model_key}")
    print(f"Loading test:     s3://{BUCKET}/{test_key}")
    print(f"Loading template: s3://{BUCKET}/{sample_key}")

    # Required files
    model_path  = download_file_from_s3(model_key, required=True, description="model")
    test_path   = download_file_from_s3(test_key, required=True, description="test data")

    # Optional sample template
    sample_path = download_file_from_s3(sample_key, required=False, description="submission template")

    # Load model and test data
    print("Loading pickled model...")
    model = joblib.load(model_path)
    print("Model loaded successfully")

    df_test = pd.read_csv(test_path)

    # Handle sample template or fallback
    if sample_path is not None:
        print("Loading sample template...")
        df_sample = pd.read_csv(sample_path)
    else:
        print("No sample template found → creating fallback from test case_ids")
        df_sample = df_test[["case_id"]].copy()
        # If your submission needs different column name (e.g. SK_ID_CURR), rename here:
        # df_sample = df_sample.rename(columns={"case_id": "SK_ID_CURR"})

    # ── Prediction ───────────────────────────────────────────────────────────────
    X_test = df_test.drop(columns=["case_id", "WEEK_NUM"], errors="ignore")
    X_test.index = df_test["case_id"]

    print("Generating predictions...")
    y_pred = model.predict_proba(X_test)[:, 1]

    df_pred = pd.DataFrame({"case_id": X_test.index, "score": y_pred})

    df_result = df_sample[["case_id"]].merge(df_pred, on="case_id", how="left")
    df_result["score"] = df_result["score"].fillna(0.005)

    # ── Save & upload ────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    local_path = os.path.join(output_dir, "sample_suggestions.csv")
    df_result.to_csv(local_path, index=False)

    gold_key = "home-credit/gold/sample_suggestions.csv"
    s3.upload_file(local_path, BUCKET, gold_key)

    print(f"✓ Predictions saved and uploaded to: s3://{BUCKET}/{gold_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/evaluation")
    args = parser.parse_args()

    evaluate_and_update(args.output_dir)