import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold

from utils.feature_utils import Pipeline  # only for column names if needed


class VotingModel:
    def __init__(self, estimators):
        self.estimators = estimators

    def predict(self, X):
        import numpy as np
        y_preds = [est.predict(X) for est in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X):
        import numpy as np
        y_preds = [est.predict_proba(X) for est in self.estimators]
        return np.mean(y_preds, axis=0)


def main():
    bucket = os.environ.get("BUCKET", "sg-home-credit")
    silver_prefix = os.environ.get("SILVER_PREFIX", "home-credit/silver")

    # In SageMaker training, you'd typically mount S3 as input channel.
    # For simplicity, read directly from S3:
    train_csv = f"s3://{bucket}/{silver_prefix}/train/train.csv"
    df_train = pd.read_csv(train_csv)

    X = df_train.drop(columns=["target", "case_id", "WEEK_NUM"])
    y = df_train["target"]
    weeks = df_train["WEEK_NUM"]

    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "colsample_bytree": 0.8,
        "colsample_bynode": 0.8,
        "verbose": -1,
        "random_state": 42,
        "device": "gpu",
    }

    cv = StratifiedGroupKFold(n_splits=5, shuffle=False)
    models = []

    for idx_train, idx_valid in cv.split(X, y, groups=weeks):
        X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
        X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.log_evaluation(100), lgb.early_stopping(100)],
        )
        models.append(model)

    voting_model = VotingModel(models)

    os.makedirs("/opt/ml/model", exist_ok=True)
    joblib.dump(voting_model, "/opt/ml/model/voting_model.pkl")


if __name__ == "__main__":
    main()
