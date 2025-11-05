"""
train_script.py
Train a RandomForestRegressor on NASA C-MAPSS FD001 data, evaluate, and log to MLflow (DagsHub).
No dagshub.init is used; tracking/auth come from environment variables or defaults.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import joblib


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA C-MAPSS asymmetric scoring function (lower is better)."""
    d = y_pred - y_true
    score = 0.0
    for di in d:
        if di < 0:
            score += np.exp(-di / 13.0) - 1.0
        else:
            score += np.exp(di / 10.0) - 1.0
    return float(score)


def load_cmapps_fd001(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load FD001 train/test/RUL and compute train RUL, mirroring app.py."""
    train_path = base_dir / "CMaps" / "train_FD001.txt"
    test_path = base_dir / "CMaps" / "test_FD001.txt"
    rul_path = base_dir / "CMaps" / "RUL_FD001.txt"

    # Raw read (space separated, many trailing blanks)
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, engine="python")
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, engine="python")
    truth_df = pd.read_csv(rul_path, sep=r"\s+", header=None, engine="python")

    # Drop trailing empty columns if any (consistent with app.py dropping 26,27)
    train_df.drop(columns=[26, 27], inplace=True, errors="ignore")
    test_df.drop(columns=[26, 27], inplace=True, errors="ignore")
    truth_df.drop(columns=[1], inplace=True, errors="ignore")

    column_names = [
        "engine_id", "time_in_cycles", "setting_1", "setting_2", "setting_3",
        "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5", "sensor_6",
        "sensor_7", "sensor_8", "sensor_9", "sensor_10", "sensor_11", "sensor_12",
        "sensor_13", "sensor_14", "sensor_15", "sensor_16", "sensor_17", "sensor_18",
        "sensor_19", "sensor_20", "sensor_21",
    ]
    train_df.columns = column_names
    test_df.columns = column_names
    truth_df.columns = ["RUL"]

    # Compute RUL for training rows
    max_cycles_df = train_df.groupby("engine_id")["time_in_cycles"].max().reset_index()
    max_cycles_df.columns = ["engine_id", "max_cycles"]
    train_df = pd.merge(train_df, max_cycles_df, on="engine_id", how="left")
    train_df["RUL"] = train_df["max_cycles"] - train_df["time_in_cycles"]
    train_df.drop(columns=["max_cycles"], inplace=True)

    return train_df, test_df, truth_df, max_cycles_df


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Match app.py: remove constant/low-corr cols and scale features."""
    # Remove constant columns
    stats = train_df.describe().transpose()
    constant_cols = stats[stats["std"] == 0].index.tolist()

    # Low correlation columns from app.py
    low_corr_cols = ["sensor_1", "sensor_5", "sensor_10", "sensor_16"]
    cols_to_drop = list(set(constant_cols + low_corr_cols))

    train_df_clean = train_df.drop(columns=cols_to_drop, errors="ignore").copy()
    test_df_clean = test_df.drop(columns=cols_to_drop, errors="ignore").copy()

    scaler = MinMaxScaler()
    feature_cols = train_df_clean.columns.drop(["engine_id", "time_in_cycles", "RUL"]).tolist()

    train_df_clean[feature_cols] = scaler.fit_transform(train_df_clean[feature_cols])
    test_df_clean[feature_cols] = scaler.transform(test_df_clean[feature_cols])

    # Optional clipping like app.py
    train_df_clean["RUL"] = train_df_clean["RUL"].clip(upper=125)

    return train_df_clean, test_df_clean, feature_cols, scaler


def main():
    load_dotenv()

    # Tracking/auth: prefer env; fallback to your repo URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/manavpatel571/mlops-project.mlflow")
    mlflow.set_tracking_uri(tracking_uri)

    # If you pass secrets via env (recommended on CI):
    # - MLFLOW_TRACKING_USERNAME
    # - MLFLOW_TRACKING_PASSWORD  (or token)
    # Nothing else required.

    project_root = Path(__file__).resolve().parent
    train_df, test_df, truth_df, _ = load_cmapps_fd001(project_root)
    train_clean, test_clean, feature_cols, scaler = preprocess(train_df, test_df)

    # Prepare a simple supervised split from training data
    X = train_clean[feature_cols].values
    y = train_clean["RUL"].values
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    params = dict(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    model = RandomForestRegressor(**params)

    mlflow.set_experiment("NASA-RUL")
    with mlflow.start_run(run_name="rf-cmaps-fd001"):
        # Log params
        mlflow.log_param("model", "RandomForestRegressor")
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Train
        model.fit(X_tr, y_tr)

        # Validate
        y_pred_val = model.predict(X_val)
        rmse_val = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
        r2_val = float(r2_score(y_val, y_pred_val))
        nasa_val = nasa_score(y_val, y_pred_val)

        mlflow.log_metric("rmse_val", rmse_val)
        mlflow.log_metric("r2_val", r2_val)
        mlflow.log_metric("nasa_score_val", nasa_val)

        # Optional: evaluate on test snapshot vs truth_df if you aggregate last cycle per engine.
        # For simplicity we log validation metrics which are consistent and fast.

        # Artifacts
        joblib.dump(model, "rf_cmaps_fd001.pkl")
        mlflow.log_artifact("rf_cmaps_fd001.pkl", artifact_path="model")

        # Save scaler and feature columns for reproducibility
        joblib.dump(scaler, "scaler.pkl")
        mlflow.log_artifact("scaler.pkl", artifact_path="preprocessing")
        with open("feature_cols.txt", "w", encoding="utf-8") as f:
            for c in feature_cols:
                f.write(f"{c}\n")
        mlflow.log_artifact("feature_cols.txt", artifact_path="preprocessing")

    print(f"✅ Logged run. RMSE(val)={rmse_val:.2f}, R2(val)={r2_val:.3f}, NASA(val)={nasa_val:.2f}")


if __name__ == "__main__":
    main()