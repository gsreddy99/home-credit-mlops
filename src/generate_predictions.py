# filename: src/generate_predictions.py

import os
import argparse
import boto3
import joblib
import pandas as pd


def main(bucket: str):
    s3 = boto3.client("s3")

    # 1) Load model from S3
    model_obj = s3.get_object(
        Bucket=bucket,
        Key="model/aiml_model.pkl"
    )
    model = joblib.load(model_obj["Body"])

    # 2) Load df_test from silver
    df_test_obj = s3.get_object(
        Bucket=bucket,
        Key="home-credit/silver/test/df_test.csv"
    )
    df_test = pd.read_csv(df_test_obj["Body"])

    # 3) Load sample_submission.csv from S3
    sample_obj = s3.get_object(
        Bucket=bucket,
        Key="model/sample_submission.csv"
    )
    df_subm = pd.read_csv(sample_obj["Body"]).set_index("case_id")

    # 4) Your notebook logic
    X_test = df_test.drop(columns=["WEEK_NUM"]).set_index("case_id")
    y_pred = model.predict_proba(X_test)[:, 1]

    df_subm["score"] = y_pred

    # 5) Save locally
    output_dir = "/opt/ml/processing/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "df_subm.csv")
    df_subm.to_csv(output_path)

    # 6) Upload to gold bucket
    s3.upload_file(
        output_path,
        bucket,
        "home-credit/gold/predictions/df_subm.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()
    main(args.bucket)
