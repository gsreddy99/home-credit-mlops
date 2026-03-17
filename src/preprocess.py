# filename: src/preprocess.py

import os
import argparse
import pandas as pd


def main(bucket: str):
    # Here you plug in your existing feature engineering logic
    # and end with a pandas DataFrame named df_test.

    # Placeholder: replace with your real df_test creation
    # df_test = <your feature engineering output>
    raise NotImplementedError("Plug your existing df_test creation logic here")

    os.makedirs("/opt/ml/processing/test", exist_ok=True)
    df_test.to_csv("/opt/ml/processing/test/df_test.csv", index=False)

    # If you also write train data, do it similarly:
    # os.makedirs("/opt/ml/processing/train", exist_ok=True)
    # df_train.to_csv("/opt/ml/processing/train/train.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()
    main(args.bucket)
