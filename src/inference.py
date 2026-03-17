# filename: inference.py

import os
import json
import boto3
import joblib
import pandas as pd

s3 = boto3.client("s3")

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "aiml_model.pkl"))
    sample = pd.read_csv(os.path.join(model_dir, "sample_submission.csv")).set_index("case_id")
    return {"model": model, "sample": sample}

def input_fn(request_body, request_content_type):
    return None  # endpoint does not accept input

def predict_fn(_, artifacts):
    model = artifacts["model"]
    sample = artifacts["sample"].copy()

    bucket = os.environ["BUCKET"]
    key = "home-credit/silver/test/df_test.csv"

    obj = s3.get_object(Bucket=bucket, Key=key)
    df_test = pd.read_csv(obj["Body"])

    X_test = df_test.drop(columns=["WEEK_NUM"]).set_index("case_id")
    preds = model.predict_proba(X_test)[:, 1]

    sample["score"] = preds
    return sample.reset_index().to_dict(orient="records")

def output_fn(predictions, content_type):
    return json.dumps({"predictions": predictions})
