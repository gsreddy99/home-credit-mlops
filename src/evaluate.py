import os
import json
import joblib
import pandas as pd
import boto3
from io import BytesIO

def load_from_s3(bucket, key):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return BytesIO(obj["Body"].read())

def main():
    bucket = os.environ["BUCKET"]

    # Load existing model from S3
    model_key = "home-credit/model/aiml_model.pkl"
    model_buf = load_from_s3(bucket, model_key)
    model = joblib.load(model_buf)

    # Load preprocessed test CSV
    test_path = f"s3://{bucket}/home-credit/silver/test/test.csv"
    df_test = pd.read_csv(test_path)

    X_test = df_test.drop(columns=["WEEK_NUM"])
    X_test = X_test.set_index("case_id")

    y_pred = model.predict_proba(X_test)[:, 1]

    # Load sample submission from S3
    sub_key = "home-credit/model/sample_submission.csv"
    sub_buf = load_from_s3(bucket, sub_key)
    df_subm = pd.read_csv(sub_buf).set_index("case_id")

    df_subm["score"] = y_pred

    # Write metrics.json
    metrics = {
        "mean_score": float(y_pred.mean()),
        "max_score": float(y_pred.max()),
        "min_score": float(y_pred.min()),
    }

    os.makedirs("/opt/ml/processing/evaluation", exist_ok=True)
    with open("/opt/ml/processing/evaluation/metrics.json", "w") as f:
        json.dump(metrics, f)

    print("Evaluation complete.")

if __name__ == "__main__":
    main()
