# src/preprocess.py
import subprocess
subprocess.run(["pip", "install", "polars", "lightgbm"], check=True)

import argparse
import os
import boto3
import polars as pl
import gc


# -------------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------------

class Pipeline:
    @staticmethod
    def set_table_dtypes(df: pl.DataFrame) -> pl.DataFrame:
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int32))
            elif col == "date_decision":
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col.endswith(("P", "A")):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col.endswith("M"):
                df = df.with_columns(pl.col(col).cast(pl.Utf8))
            elif col.endswith("D"):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    @staticmethod
    def handle_dates(df: pl.DataFrame) -> pl.DataFrame:
        date_cols = [c for c in df.columns if c.endswith("D") and c != "date_decision"]
        if date_cols:
            df = df.with_columns([
                (pl.col(c) - pl.col("date_decision")).dt.total_days().cast(pl.Float32).alias(c)
                for c in date_cols
            ])
        return df.drop(["date_decision", "MONTH"], strict=False)

    @staticmethod
    def filter_cols(df: pl.DataFrame) -> pl.DataFrame:
        keep = {"target", "case_id", "WEEK_NUM"}
        to_drop = []
        for col in df.columns:
            if col in keep:
                continue
            null_rate = df[col].is_null().mean()
            if null_rate > 0.96:
                to_drop.append(col)
                continue
            if df[col].dtype == pl.Utf8:
                n_unique = df[col].n_unique()
                if n_unique <= 1 or n_unique > 200:
                    to_drop.append(col)
        if to_drop:
            df = df.drop(to_drop)
        return df


class Aggregator:
    @staticmethod
    def num_expr(df):   return [pl.col(c).max().alias(f"max_{c}") for c in df.columns if c.endswith(("P","A"))]
    @staticmethod
    def date_expr(df):  return [pl.col(c).max().alias(f"max_{c}") for c in df.columns if c.endswith("D")]
    @staticmethod
    def str_expr(df):   return [pl.col(c).max().alias(f"max_{c}") for c in df.columns if c.endswith("M")]
    @staticmethod
    def other_expr(df): return [pl.col(c).max().alias(f"max_{c}") for c in df.columns if c.endswith(("T","L"))]
    @staticmethod
    def count_expr(df): return [pl.col(c).max().alias(f"max_{c}") for c in df.columns if "num_group" in c]

    @staticmethod
    def get_exprs(df):
        return (
            Aggregator.num_expr(df) +
            Aggregator.date_expr(df) +
            Aggregator.str_expr(df) +
            Aggregator.other_expr(df) +
            Aggregator.count_expr(df)
        )


# -------------------------------------------------------------------------
#  S3-aware file loaders (fixes glob issue)
# -------------------------------------------------------------------------

def list_s3_parquet(bucket, prefix):
    s3 = boto3.client("s3")
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in resp:
        return []
    return [obj["Key"] for obj in resp["Contents"] if obj["Key"].endswith(".parquet")]


def read_file(s3_path: str, depth=None) -> pl.DataFrame:
    df = pl.read_parquet(s3_path).pipe(Pipeline.set_table_dtypes)
    if depth in (1, 2):
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
    return df


def read_files(bucket: str, prefix: str, depth=None) -> pl.DataFrame:
    keys = list_s3_parquet(bucket, prefix)
    if not keys:
        print(f"Warning: No files found for prefix {prefix}")
        return pl.DataFrame()

    dfs = [read_file(f"s3://{bucket}/{k}", depth) for k in keys]
    return pl.concat(dfs, how="vertical_relaxed").unique("case_id")


# -------------------------------------------------------------------------
#  Feature Engineering
# -------------------------------------------------------------------------

def feature_engineering(df_base, depth_0, depth_1, depth_2):
    df = df_base.with_columns([
        pl.col("date_decision").dt.month().alias("month_decision"),
        pl.col("date_decision").dt.weekday().alias("weekday_decision"),
    ])

    # Unique suffix per join to avoid DuplicateError
    for group_idx, group in enumerate([depth_0, depth_1, depth_2]):
        for table_idx, df_depth in enumerate(group):
            if df_depth is not None and df_depth.height > 0:
                suffix = f"_d{group_idx}_{table_idx}"
                df = df.join(df_depth, on="case_id", how="left", suffix=suffix)

    df = Pipeline.handle_dates(df)
    return df


def to_pandas(df: pl.DataFrame):
    pdf = df.to_pandas()  # works with Pandas 1.1.3
    cat_cols = pdf.select_dtypes(include=["object", "category"]).columns.tolist()
    pdf[cat_cols] = pdf[cat_cols].astype("category")
    return pdf, cat_cols


# -------------------------------------------------------------------------
#  Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, required=True)
    parser.add_argument("--train-prefix", type=str, default="home-credit/bronze/train")
    parser.add_argument("--test-prefix",  type=str, default="home-credit/bronze/test")
    args = parser.parse_args()

    bucket = args.bucket

    # --------------------------
    # TRAIN
    # --------------------------
    train_base = read_file(f"s3://{bucket}/{args.train_prefix}/train_base.parquet")

    depth2_prefix = f"{args.train_prefix}/train_credit_bureau_a_2_"

    train_store = {
        "df_base": train_base,
        "depth_0": [
            read_file(f"s3://{bucket}/{args.train_prefix}/train_static_cb_0.parquet"),
        ],
        "depth_1": [
            read_file(f"s3://{bucket}/{args.train_prefix}/train_applprev_1_0.parquet", 1),
            read_file(f"s3://{bucket}/{args.train_prefix}/train_credit_bureau_b_1.parquet", 1),
            read_file(f"s3://{bucket}/{args.train_prefix}/train_tax_registry_a_1.parquet", 1),
            read_file(f"s3://{bucket}/{args.train_prefix}/train_tax_registry_b_1.parquet", 1),
            read_file(f"s3://{bucket}/{args.train_prefix}/train_tax_registry_c_1.parquet", 1),
            read_file(f"s3://{bucket}/{args.train_prefix}/train_person_1.parquet", 1),
            read_file(f"s3://{bucket}/{args.train_prefix}/train_debitcard_1.parquet", 1),
            read_file(f"s3://{bucket}/{args.train_prefix}/train_deposit_1.parquet", 1),
            read_file(f"s3://{bucket}/{args.train_prefix}/train_other_1.parquet", 1),
        ],
        "depth_2": [
            read_files(bucket, depth2_prefix, 2),
            read_file(f"s3://{bucket}/{args.train_prefix}/train_credit_bureau_b_2.parquet", 2),
            read_file(f"s3://{bucket}/{args.train_prefix}/train_applprev_2.parquet", 2),
            read_file(f"s3://{bucket}/{args.train_prefix}/train_person_2.parquet", 2),
        ]
    }

    print("Starting train feature engineering...")
    df_train = feature_engineering(**train_store)
    df_train = Pipeline.filter_cols(df_train)
    df_train_pd, cat_cols = to_pandas(df_train)

    # --------------------------
    # TEST
    # --------------------------
    test_base = read_file(f"s3://{bucket}/{args.test_prefix}/test_base.parquet")

    test_depth2_prefix = f"{args.test_prefix}/test_credit_bureau_a_2_"

    test_store = {
        "df_base": test_base,
        "depth_0": [
            read_file(f"s3://{bucket}/{args.test_prefix}/test_static_0_0.parquet"),
            read_file(f"s3://{bucket}/{args.test_prefix}/test_static_0_1.parquet"),
            read_file(f"s3://{bucket}/{args.test_prefix}/test_static_0_2.parquet"),
            read_file(f"s3://{bucket}/{args.test_prefix}/test_static_cb_0.parquet"),
        ],
        "depth_1": [
            read_file(f"s3://{bucket}/{args.test_prefix}/test_applprev_1_0.parquet", 1),
            read_file(f"s3://{bucket}/{args.test_prefix}/test_credit_bureau_b_1.parquet", 1),
            read_file(f"s3://{bucket}/{args.test_prefix}/test_tax_registry_a_1.parquet", 1),
            read_file(f"s3://{bucket}/{args.test_prefix}/test_tax_registry_b_1.parquet", 1),
            read_file(f"s3://{bucket}/{args.test_prefix}/test_tax_registry_c_1.parquet", 1),
            read_file(f"s3://{bucket}/{args.test_prefix}/test_person_1.parquet", 1),
            read_file(f"s3://{bucket}/{args.test_prefix}/test_debitcard_1.parquet", 1),
            read_file(f"s3://{bucket}/{args.test_prefix}/test_deposit_1.parquet", 1),
            read_file(f"s3://{bucket}/{args.test_prefix}/test_other_1.parquet", 1),
            read_files(bucket, f"{args.test_prefix}/test_credit_bureau_a_1_", 1),
        ],
        "depth_2": [
            read_files(bucket, test_depth2_prefix, 2),
            read_file(f"s3://{bucket}/{args.test_prefix}/test_credit_bureau_b_2.parquet", 2),
            read_file(f"s3://{bucket}/{args.test_prefix}/test_person_2.parquet", 2),
        ]
    }

    print("Starting test feature engineering...")
    df_test = feature_engineering(**test_store)

    common_cols = [c for c in df_train.columns if c != "target"]
    df_test = df_test.select([c for c in common_cols if c in df_test.columns])
    df_test_pd, _ = to_pandas(df_test)

    # --------------------------
    # Save & Upload
    # --------------------------
    os.makedirs("/opt/ml/processing/train", exist_ok=True)
    os.makedirs("/opt/ml/processing/test", exist_ok=True)

    train_out = "/opt/ml/processing/train/train.csv"
    test_out  = "/opt/ml/processing/test/test.csv"

    df_train_pd.to_csv(train_out, index=False)
    df_test_pd.to_csv(test_out, index=False)

    s3 = boto3.client("s3")
    s3.upload_file(train_out, bucket, "home-credit/silver/train/train.csv")
    s3.upload_file(test_out,  bucket, "home-credit/silver/test/test.csv")

    print("Preprocessing completed.")
    print(f"Train shape: {df_train_pd.shape}")
    print(f"Test shape:  {df_test_pd.shape}")


if __name__ == "__main__":
    main()
