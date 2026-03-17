import os
import io
from glob import glob

import boto3
import polars as pl
import pandas as pd

from utils.feature_utils import Pipeline, Aggregator, feature_eng

s3 = boto3.client("s3")


def prefix_exists(bucket: str, prefix: str) -> bool:
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return "Contents" in resp


def save_csv_to_s3(df: pd.DataFrame, bucket: str, key: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def read_file(path: str, depth=None) -> pl.DataFrame:
    df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)
    if depth in [1, 2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
    return df


def read_files(regex_path: str, depth=None) -> pl.DataFrame:
    chunks = []
    for path in glob(regex_path):
        df = pl.read_parquet(path)
        df = df.pipe(Pipeline.set_table_dtypes)
        if depth in [1, 2]:
            df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
        chunks.append(df)
    df = pl.concat(chunks, how="vertical_relaxed")
    return df.unique(subset=["case_id"])


def to_pandas(df_data: pl.DataFrame) -> pd.DataFrame:
    df_data = df_data.to_pandas()
    cat_cols = list(df_data.select_dtypes("object").columns)
    if cat_cols:
        df_data[cat_cols] = df_data[cat_cols].astype("category")
    return df_data


def main():
    bucket = os.environ.get("BUCKET", "sg-home-credit")
    bronze_prefix = os.environ.get("BRONZE_PREFIX", "home-credit/bronze")
    silver_prefix = os.environ.get("SILVER_PREFIX", "home-credit/silver")

    train_base = f"s3://{bucket}/{bronze_prefix}/train/train_base.parquet"
    test_base = f"s3://{bucket}/{bronze_prefix}/test/test_base.parquet"

    # ---- TRAIN ----
    train_dir = f"s3://{bucket}/{bronze_prefix}/train/"

    train_data_store = {
        "df_base": read_file(train_base),
        "depth_0": [
            read_file(f"{train_dir}train_static_cb_0.parquet"),
            read_files(f"{train_dir}train_static_0_*.parquet"),
        ],
        "depth_1": [
            read_files(f"{train_dir}train_applprev_1_*.parquet", 1),
            read_file(f"{train_dir}train_tax_registry_a_1.parquet", 1),
            read_file(f"{train_dir}train_tax_registry_b_1.parquet", 1),
            read_file(f"{train_dir}train_tax_registry_c_1.parquet", 1),
            read_files(f"{train_dir}train_credit_bureau_a_1_*.parquet", 1),
            read_file(f"{train_dir}train_credit_bureau_b_1.parquet", 1),
            read_file(f"{train_dir}train_other_1.parquet", 1),
            read_file(f"{train_dir}train_person_1.parquet", 1),
            read_file(f"{train_dir}train_deposit_1.parquet", 1),
            read_file(f"{train_dir}train_debitcard_1.parquet", 1),
        ],
        "depth_2": [
            read_file(f"{train_dir}train_credit_bureau_b_2.parquet", 2),
            read_files(f"{train_dir}train_credit_bureau_a_2_*.parquet", 2),
        ],
    }

    df_train = feature_eng(**train_data_store)
    df_train = df_train.pipe(Pipeline.filter_cols)
    df_train = to_pandas(df_train)

    silver_train_prefix = f"{silver_prefix}/train/"
    if not prefix_exists(bucket, silver_train_prefix):
        s3.put_object(Bucket=bucket, Key=silver_train_prefix)

    save_csv_to_s3(df_train, bucket, f"{silver_prefix}/train/train.csv")

    # ---- TEST ----
    test_dir = f"s3://{bucket}/{bronze_prefix}/test/"

    test_data_store = {
        "df_base": read_file(test_base),
        "depth_0": [
            read_file(f"{test_dir}test_static_cb_0.parquet"),
            read_files(f"{test_dir}test_static_0_*.parquet"),
        ],
        "depth_1": [
            read_files(f"{test_dir}test_applprev_1_*.parquet", 1),
            read_file(f"{test_dir}test_tax_registry_a_1.parquet", 1),
            read_file(f"{test_dir}test_tax_registry_b_1.parquet", 1),
            read_file(f"{test_dir}test_tax_registry_c_1.parquet", 1),
            read_files(f"{test_dir}test_credit_bureau_a_1_*.parquet", 1),
            read_file(f"{test_dir}test_credit_bureau_b_1.parquet", 1),
            read_file(f"{test_dir}test_other_1.parquet", 1),
            read_file(f"{test_dir}test_person_1.parquet", 1),
            read_file(f"{test_dir}test_deposit_1.parquet", 1),
            read_file(f"{test_dir}test_debitcard_1.parquet", 1),
        ],
        "depth_2": [
            read_file(f"{test_dir}test_credit_bureau_b_2.parquet", 2),
            read_files(f"{test_dir}test_credit_bureau_a_2_*.parquet", 2),
        ],
    }

    df_test = feature_eng(**test_data_store)
    # align columns with train (drop target if present)
    df_test = df_test.select([c for c in df_train.columns if c != "target"])
    df_test = to_pandas(df_test)

    silver_test_prefix = f"{silver_prefix}/test/"
    if not prefix_exists(bucket, silver_test_prefix):
        s3.put_object(Bucket=bucket, Key=silver_test_prefix)

    save_csv_to_s3(df_test, bucket, f"{silver_prefix}/test/test.csv")


if __name__ == "__main__":
    main()
