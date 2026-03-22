# filename: src/train.py

import os
import sys
import subprocess

# --------------------------------------------------------
# Install dependencies (same pattern as evaluate.py)
# --------------------------------------------------------
def prepare_environment():
    req_path = "/opt/ml/processing/input/reqs/requirements.txt"
    if os.path.exists(req_path):
        print("Installing training dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--no-cache-dir",
            "-r", req_path
        ])
        print("✓ Training environment ready.")

prepare_environment()

# --------------------------------------------------------
# Standard imports AFTER dependency installation
# --------------------------------------------------------
import argparse
import pandas as pd
import joblib
import boto3
import tempfile
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold

BUCKET = "sg-home-credit"


def download_file_from_s3(key):
    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        s3.download_file(BUCKET, key, tmp.name)
        return tmp.name


def train_model(model_output_path):

    # --------------------------------------------------------
    # 1. Download train.csv from S3 silver layer
    # --------------------------------------------------------
    train_key = "home-credit/silver/train/train.csv"
    print(f"Loading train.csv from s3://{BUCKET}/{train_key}")

    train_path = download_file_from_s3(train_key)
    df = pd.read_csv(train_path)
    os.unlink(train_path)

    # --------------------------------------------------------
    # 2. Prepare data
    # --------------------------------------------------------
    X = df.drop(columns=["target", "case_id", "WEEK_NUM"], errors="ignore")
    y = df["target"]
    groups = df["WEEK_NUM"]

    # Convert object columns to category (LightGBM requirement)
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category")

    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "colsample_bytree": 0.8,
        "colsample_bynode": 0.8,
        "verbose": -1,
        "random_state": 42,
        "device": "gpu",
    }

    cv = StratifiedGroupKFold(n_splits=5, shuffle=False)
    models = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups=groups)):
        print(f"Training fold {fold+1}/5...")
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[valid_idx], y.iloc[valid_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.log_evaluation(100),
                lgb.early_stopping(stopping_rounds=100, verbose=True)
            ]
        )
        models.append(model)

    # --------------------------------------------------------
    # 3. Create voting ensemble
    # --------------------------------------------------------
    class VotingModel:
        def __init__(self, estimators):
            self.estimators = estimators

        def predict_proba(self, X):
            probs = [est.predict_proba(X) for est in self.estimators]
            return sum(probs) / len(probs)

    final_model = VotingModel(models)

    # --------------------------------------------------------
    # 4. Save model
    # --------------------------------------------------------
    os.makedirs(model_output_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_filename = f"model_{timestamp}.pkl"
    latest_filename = "aiml_model.pkl"   # match evaluate.py

    local_timestamped_path = os.path.join(model_output_path, timestamped_filename)
    local_latest_path = os.path.join(model_output_path, latest_filename)

    joblib.dump(final_model, local_timestamped_path)
    joblib.dump(final_model, local_latest_path)

    print(f"✓ Timestamped model saved to {local_timestamped_path}")
    print(f"✓ Latest model saved to {local_latest_path}")

    # --------------------------------------------------------
    # 5. Upload to S3
    # --------------------------------------------------------
    s3 = boto3.client("s3")

    timestamped_key = f"home-credit/model/archive/{timestamped_filename}"
    s3.upload_file(local_timestamped_path, BUCKET, timestamped_key)
    print(f"✓ Uploaded timestamped model to s3://{BUCKET}/{timestamped_key}")

    latest_key = "home-credit/model/aiml_model.pkl"
    s3.upload_file(local_latest_path, BUCKET, latest_key)
    print(f"✓ Uploaded latest model to s3://{BUCKET}/{latest_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output", type=str, default="/opt/ml/processing/model")
    args = parser.parse_args()

    train_model(args.model_output)
