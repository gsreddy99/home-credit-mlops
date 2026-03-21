# filename: src/evaluate.py

import os
import argparse
import sys
import subprocess
import boto3
import pandas as pd
import joblib
import tempfile
import importlib


BUCKET = "sg-home-credit"


def run_pip_install(package_spec, upgrade_only=False):
    """Install or upgrade a package at runtime"""
    flag = "--upgrade" if upgrade_only else "--force-reinstall"
    print(f"Running pip install {flag} {package_spec}")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--no-cache-dir", flag, package_spec
        ])
        print(f"Success: {package_spec}")
    except Exception as e:
        print(f"Failed to install/upgrade {package_spec}: {e}", file=sys.stderr)
        raise


def ensure_dependency(package_name, min_version=None, install_spec=None):
    """Check if package is importable and meets min version; install/upgrade if needed"""
    if install_spec is None:
        install_spec = package_name

    try:
        pkg = importlib.import_module(package_name)
        version = getattr(pkg, "__version__", "unknown")
        print(f"{package_name} version: {version}")

        if min_version and version != "unknown":
            from packaging import version as ver
            if ver.parse(version) < ver.parse(min_version):
                print(f"{package_name} {version} < {min_version} → upgrading")
                run_pip_install(f"{install_spec}>={min_version}", upgrade_only=True)
                # Re-import after upgrade
                importlib.invalidate_caches()
                pkg = importlib.import_module(package_name)
                print(f"Upgraded {package_name} to {getattr(pkg, '__version__', 'unknown')}")
        return pkg

    except ImportError:
        print(f"{package_name} not found → installing")
        run_pip_install(install_spec)
        importlib.invalidate_caches()
        return importlib.import_module(package_name)

    except Exception as e:
        print(f"Unexpected error checking {package_name}: {e}", file=sys.stderr)
        raise


def prepare_environment():
    """Ensure all likely required packages are present and compatible"""
    print("Checking / preparing runtime environment...")

    # Order matters somewhat: numpy first (affects many others)
    ensure_dependency("numpy", min_version="2.0.0")

    # Then lightgbm (your model core)
    ensure_dependency("lightgbm", min_version="4.0.0")

    # scikit-learn – very common for lightgbm sklearn API
    ensure_dependency("sklearn", install_spec="scikit-learn", min_version="1.0.0")

    # scipy – often pulled by lightgbm / sklearn
    ensure_dependency("scipy")

    # pandas – already used heavily in script
    ensure_dependency("pandas", min_version="1.5.0")

    print("Environment preparation finished.")


def download_file_from_s3(key, required=True, description="file"):
    s3 = boto3.client("s3")
    full_path = f"s3://{BUCKET}/{key}"
    print(f"Downloading {description}: {full_path}")

    try:
        s3.head_object(Bucket=BUCKET, Key=key)
    except s3.exceptions.ClientError as e:
        if e.response.get('Error', {}).get('Code') == '404':
            msg = f"File not found: {full_path}"
            if required:
                print(msg, file=sys.stderr)
                raise FileNotFoundError(msg)
            else:
                print(msg + " → using fallback")
                return None
        else:
            raise

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        s3.download_file(BUCKET, key, tmp.name)
        print(f"Downloaded: {full_path}")
        return tmp.name


class VotingModel:
    def __init__(self, estimators):
        self.estimators = estimators

    def predict_proba(self, X):
        probs = [est.predict_proba(X) for est in self.estimators]
        return sum(probs) / len(probs)


def evaluate_and_update(output_dir: str):
    # Prepare dependencies first
    prepare_environment()

    s3 = boto3.client("s3")

    # ── Input paths ──────────────────────────────────────────────────────────────
    model_key   = "home-credit/model/aiml_model.pkl"
    test_key    = "home-credit/silver/test/test.csv"
    sample_key  = "home-credit/model/sample_suggestions.csv"   # ← change if actually sample_submission.csv

    print(f"Model:    s3://{BUCKET}/{model_key}")
    print(f"Test:     s3://{BUCKET}/{test_key}")
    print(f"Template: s3://{BUCKET}/{sample_key}")

    model_path  = download_file_from_s3(model_key, required=True, description="model")
    test_path   = download_file_from_s3(test_key, required=True, description="test")

    sample_path = download_file_from_s3(sample_key, required=False, description="submission template")

    # Load model
    print("Loading model...")
    model = joblib.load(model_path)
    print("Model loaded successfully")

    df_test = pd.read_csv(test_path)

    if sample_path is not None:
        print("Loading template...")
        df_sample = pd.read_csv(sample_path)
    else:
        print("No template found → fallback from test case_ids")
        df_sample = df_test[["case_id"]].copy()

    # ── Predict ────────────────────────────────────────────────────────────────
    X_test = df_test.drop(columns=["case_id", "WEEK_NUM"], errors="ignore")
    X_test.index = df_test["case_id"]

    print("Generating predictions...")
    y_pred = model.predict_proba(X_test)[:, 1]

    df_pred = pd.DataFrame({"case_id": X_test.index, "score": y_pred})

    df_result = df_sample[["case_id"]].merge(df_pred, on="case_id", how="left")
    df_result["score"] = df_result["score"].fillna(0.005)

    # ── Save & upload ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    local_path = os.path.join(output_dir, "sample_suggestions.csv")
    df_result.to_csv(local_path, index=False)

    gold_key = "home-credit/gold/sample_suggestions.csv"
    s3.upload_file(local_path, BUCKET, gold_key)

    print(f"✓ Uploaded to: s3://{BUCKET}/{gold_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/evaluation")
    args = parser.parse_args()

    evaluate_and_update(args.output_dir)