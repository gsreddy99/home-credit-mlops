import os
import boto3
import polars as pl
from io import BytesIO
from utils.feature_utils import Pipeline, Aggregator

# -----------------------------
# S3 Helpers
# -----------------------------
def read_parquet_s3_lazy(bucket, key):
    return pl.scan_parquet(f"s3://{bucket}/{key}")

def list_s3_files(bucket, prefix):
    s3 = boto3.client("s3")
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [c["Key"] for c in resp.get("Contents", []) if c["Key"].endswith(".parquet")]

# -----------------------------
# Depth-based loaders (lazy)
# -----------------------------
def load_depth_group_lazy(bucket, files, depth=None):
    dfs = []
    for f in files:
        df = read_parquet_s3_lazy(bucket, f)
        df = df.pipe(Pipeline.set_table_dtypes)

        if depth in [1, 2]:
            df = df.groupby("case_id").agg(Aggregator.get_exprs(df))

        dfs.append(df)
    return dfs

# -----------------------------
# Feature Engineering (lazy)
# -----------------------------
def feature_eng_lazy(df_base, depth_0, depth_1, depth_2):
    df = df_base.with_columns([
        pl.col("date_decision").dt.month().alias("month_decision"),
        pl.col("date_decision").dt.weekday().alias("weekday_decision"),
    ])

    for i, d in enumerate(depth_0 + depth_1 + depth_2):
        df = df.join(d, on="case_id", how="left", suffix=f"_{i}")

    df = df.pipe(Pipeline.handle_dates)
    return df

# -----------------------------
# Main Preprocess
# -----------------------------
def main():
    bucket = os.environ["BUCKET"]

    # -------------------------
    # TRAIN FILE GROUPING
    # -------------------------
    train_prefix = "home-credit/bronze/train"

    train_base = f"{train_prefix}/train_base.parquet"

    depth0_train = [
        f"{train_prefix}/train_static_cb_0.parquet",
        f"{train_prefix}/train_person_1.parquet",
        f"{train_prefix}/train_person_2.parquet",
        f"{train_prefix}/train_other_1.parquet",
        f"{train_prefix}/train_deposit_1.parquet",
        f"{train_prefix}/train_debitcard_1.parquet",
    ]

    depth1_train = [
        f"{train_prefix}/train_applprev_1_0.parquet",
        f"{train_prefix}/train_tax_registry_a_1.parquet",
        f"{train_prefix}/train_tax_registry_b_1.parquet",
        f"{train_prefix}/train_tax_registry_c_1.parquet",
        f"{train_prefix}/train_credit_bureau_b_1.parquet",
    ]

    depth2_train = [
        f"{train_prefix}/train_applprev_2.parquet",
        f"{train_prefix}/train_credit_bureau_b_2.parquet",
    ]

    depth2_train += [
        f for f in list_s3_files(bucket, train_prefix)
        if "train_credit_bureau_a_2_" in f
    ]

    # -------------------------
    # TEST FILE GROUPING
    # -------------------------
    test_prefix = "home-credit/bronze/test"

    test_base = f"{test_prefix}/test_base.parquet"

    depth0_test = [
        f"{test_prefix}/test_static_cb_0.parquet",
        f"{test_prefix}/test_static_0_0.parquet",
        f"{test_prefix}/test_static_0_1.parquet",
        f"{test_prefix}/test_static_0_2.parquet",
        f"{test_prefix}/test_person_1.parquet",
        f"{test_prefix}/test_person_2.parquet",
        f"{test_prefix}/test_other_1.parquet",
        f"{test_prefix}/test_deposit_1.parquet",
        f"{test_prefix}/test_debitcard_1.parquet",
    ]

    depth1_test = [
        f"{test_prefix}/test_tax_registry_a_1.parquet",
        f"{test_prefix}/test_tax_registry_b_1.parquet",
        f"{test_prefix}/test_tax_registry_c_1.parquet",
        f"{test_prefix}/test_credit_bureau_b_1.parquet",
    ]

    depth1_test += [
        f for f in list_s3_files(bucket, test_prefix)
        if "test_credit_bureau_a_1_" in f
    ]

    depth2_test = [
        f"{test_prefix}/test_credit_bureau_b_2.parquet",
    ]

    depth2_test += [
        f for f in list_s3_files(bucket, test_prefix)
        if "test_credit_bureau_a_2_" in f
    ]

    # -------------------------
    # LOAD BASE FILES (lazy)
    # -------------------------
    df_train_base = read_parquet_s3_lazy(bucket, train_base).pipe(Pipeline.set_table_dtypes)
    df_test_base = read_parquet_s3_lazy(bucket, test_base).pipe(Pipeline.set_table_dtypes)

    # -------------------------
    # LOAD DEPTH GROUPS (lazy)
    # -------------------------
    depth0_train = load_depth_group_lazy(bucket, depth0_train, depth=None)
    depth1_train = load_depth_group_lazy(bucket, depth1_train, depth=1)
    depth2_train = load_depth_group_lazy(bucket, depth2_train, depth=2)

    depth0_test = load_depth_group_lazy(bucket, depth0_test, depth=None)
    depth1_test = load_depth_group_lazy(bucket, depth1_test, depth=1)
    depth2_test = load_depth_group_lazy(bucket, depth2_test, depth=2)

    # -------------------------
    # FEATURE ENGINEERING (lazy)
    # -------------------------
    df_train = feature_eng_lazy(df_train_base, depth0_train, depth1_train, depth2_train)
    df_test = feature_eng_lazy(df_test_base, depth0_test, depth1_test, depth2_test)

    # -------------------------
    # FILTER + ALIGN COLUMNS
    # -------------------------
    df_train = df_train.collect()
    df_train = Pipeline.filter_cols(df_train)

    df_test = df_test.collect()
    df_test = df_test.select([c for c in df_train.columns if c != "target"])

    # -------------------------
    # WRITE OUTPUTS
    # -------------------------
    train_out = f"s3://{bucket}/home-credit/silver/train/train.csv"
    test_out = f"s3://{bucket}/home-credit/silver/test/test.csv"

    df_train.write_csv(train_out)
    df_test.write_csv(test_out)

    print("Saved:")
    print(train_out)
    print(test_out)


if __name__ == "__main__":
    main()
