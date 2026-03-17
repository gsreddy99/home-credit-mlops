import os
import json

import boto3
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    bucket = os.environ.get("BUCKET", "sg-home-credit")
    silver_prefix = os.environ.get("SILVER_PREFIX", "home-credit/silver")
    gold_prefix = os.environ.get("GOLD_PREFIX", "home-credit/gold")
    execution_id = os.environ.get("EXECUTION_ID", "local-execution")

    # Load data
    train_csv = f"s3://{bucket}/{silver_prefix}/train/train.csv"
    df_train = pd.read_csv(train_csv)

    X = df_train.drop(columns=["target", "case_id", "WEEK_NUM"])
    y = df_train["target"]

    # Load model (in SageMaker Processing, you'd mount model artifact;
    # here we assume it's available locally)
    model_path = os.environ.get("MODEL_PATH", "/opt/ml/model/voting_model.pkl")
    model = joblib.load(model_path)

    y_pred = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred)

    s3 = boto3.client("s3")
    eval_prefix = f"{gold_prefix}/evaluation/{execution_id}/"
    # create prefix marker if needed
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=eval_prefix, MaxKeys=1)
    if "Contents" not in resp:
        s3.put_object(Bucket=bucket, Key=eval_prefix)

    key = f"{eval_prefix}evaluation.json"
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps({"auc": auc}))


if __name__ == "__main__":
    main()
