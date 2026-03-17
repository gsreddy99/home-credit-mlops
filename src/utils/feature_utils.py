import polars as pl
import numpy as np


class Pipeline:
    @staticmethod
    def set_table_dtypes(df: pl.DataFrame) -> pl.DataFrame:
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int32))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    @staticmethod
    def handle_dates(df: pl.DataFrame) -> pl.DataFrame:
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days())
                df = df.with_columns(pl.col(col).cast(pl.Float32))

        df = df.drop("date_decision", "MONTH", strict=False)
        return df

    @staticmethod
    def filter_cols(df: pl.DataFrame) -> pl.DataFrame:
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()
                if isnull > 0.95:
                    df = df.drop(col)

        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) and (df[col].dtype == pl.String):
                freq = df[col].n_unique()
                if (freq == 1) or (freq > 200):
                    df = df.drop(col)

        return df


class Aggregator:
    @staticmethod
    def num_expr(df: pl.DataFrame):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        return [pl.max(col).alias(f"max_{col}") for col in cols]

    @staticmethod
    def date_expr(df: pl.DataFrame):
        cols = [col for col in df.columns if col[-1] in ("D",)]
        return [pl.max(col).alias(f"max_{col}") for col in cols]

    @staticmethod
    def str_expr(df: pl.DataFrame):
        cols = [col for col in df.columns if col[-1] in ("M",)]
        return [pl.max(col).alias(f"max_{col}") for col in cols]

    @staticmethod
    def other_expr(df: pl.DataFrame):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        return [pl.max(col).alias(f"max_{col}") for col in cols]

    @staticmethod
    def count_expr(df: pl.DataFrame):
        cols = [col for col in df.columns if "num_group" in col]
        return [pl.max(col).alias(f"max_{col}") for col in cols]

    @staticmethod
    def get_exprs(df: pl.DataFrame):
        return (
            Aggregator.num_expr(df)
            + Aggregator.date_expr(df)
            + Aggregator.str_expr(df)
            + Aggregator.other_expr(df)
            + Aggregator.count_expr(df)
        )


def feature_eng(df_base, depth_0, depth_1, depth_2):
    df_base = (
        df_base
        .with_columns(
            month_decision=pl.col("date_decision").dt.month(),
            weekday_decision=pl.col("date_decision").dt.weekday(),
        )
    )

    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")

    df_base = Pipeline.handle_dates(df_base)
    return df_base
