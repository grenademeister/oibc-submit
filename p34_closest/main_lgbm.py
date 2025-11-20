#!/usr/bin/env python3
import logging, warnings, yaml, pickle, time, joblib
from datetime import datetime
from pathlib import Path
from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

from preprocess import engineer_features, build_feature_matrix, TARGET

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Model parameter definitions
# ---------------------------------------------------------------------
BASE_MODEL_PARAMS: Dict[str, Dict] = {
    "lightgbm": {
        "objective": "regression",
        "metric": "mae",
        "n_estimators": 4000,
        "learning_rate": 0.046349,
        "num_leaves": 200, # 110
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
}

GPU_MODEL_PARAMS: Dict[str, Dict] = {
    "lightgbm": {"device": "cuda", "gpu_use_dp": False},
}

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, "w"), logging.StreamHandler()]
    )

def check_gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        import subprocess
        try:
            subprocess.check_output(["nvidia-smi"])
            return True
        except Exception:
            return False

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setting_dir = Path(cfg["save_path"]) / "setting" / timestamp
    model_dir = Path(cfg["save_path"]) / "model"
    setting_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg["setting_dir"], cfg["model_dir"] = str(setting_dir), str(model_dir)
    with open(setting_dir / "config.yaml", "w") as f_out:
        yaml.safe_dump(cfg, f_out)
    return cfg

def build_model(use_gpu=False):
    params = BASE_MODEL_PARAMS["lightgbm"].copy()
    if use_gpu:
        params.update(GPU_MODEL_PARAMS["lightgbm"])
    return LGBMRegressor(**params)

# ---------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------
def train_single_model(gpu, X_train, X_val, y_train, y_val, cat_features, model_dir: str | Path):
    start_time = time.time()
    logging.info("▶ START training LIGHTGBM")

    model = build_model(use_gpu=gpu)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        categorical_feature=cat_features or None,
    )

    val_preds = np.maximum(model.predict(X_val), 0)

    model_path = Path(model_dir) / "lightgbm.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # ------------------------- NEW: Save feature importance -------------------------
    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    fi_csv_path = Path(model_dir) / "feature_importance.csv"
    fi.to_csv(fi_csv_path, index=False)
    logging.info(f"Saved feature importance CSV to {fi_csv_path}")

    plt.figure(figsize=(10, 12))
    plt.barh(fi["feature"], fi["importance"])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fi_png_path = Path(model_dir) / "feature_importance.png"
    plt.savefig(fi_png_path, dpi=200)
    plt.close()
    logging.info(f"Saved feature importance PNG to {fi_png_path}")
    # ------------------------------------------------------------------------------

    logging.info(f"✓ FINISHED LIGHTGBM ({time.time() - start_time:.2f}s)")
    return val_preds

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    preprocess_only = False

    cfg = load_config("config.yaml")
    log_path = Path(cfg["setting_dir"]) / "training.log"
    setup_logging(log_path)

    gpu = check_gpu_available() and cfg["use_gpu"]
    if cfg["use_gpu"] and not gpu:
        logging.warning("GPU requested but not available — running on CPU.")

    if "processed_cache_path" in cfg and cfg["processed_cache_path"]:
        processed_dir = Path(cfg["processed_cache_path"])
    else:
        processed_dir = Path(cfg["save_path"]) / "processed_features"

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train_processed.joblib"
    val_path = processed_dir / "val_processed.joblib"
    test_path = processed_dir / "test_processed.joblib"
    logging.info(f"Using processed cache path: {processed_dir}")

    logging.info("▶ Loading pre-built train/validation cluster features...")
    train_raw = joblib.load(cfg["train_cluster_features"])
    val_raw = joblib.load(cfg["val_cluster_features"])
    test_raw = joblib.load(cfg["test_cluster_features"])
    logging.info(f"✅ Loaded train {train_raw.shape}, val {val_raw.shape}, test {test_raw.shape}")

    datasets = {
        "train": (train_path, train_raw),
        "val": (val_path, val_raw),
        "test": (test_path, test_raw),
    }
    processed_results = {}

    for name, (path, raw_df) in datasets.items():
        if path.exists():
            logging.info(f"💾 Using cached {name}_processed from {path}")
            processed_results[name] = joblib.load(path)
        else:
            logging.info(f"🧠 Processing {name} features (cache not found)...")
            df_proc = engineer_features(raw_df)
            if name == "train":
                df_proc = df_proc.dropna(subset=[TARGET])
            joblib.dump(df_proc, path)
            processed_results[name] = df_proc
            logging.info(f"💾 Saved {name}_processed to {path}")

    train_processed = processed_results["train"]
    val_processed = processed_results["val"]
    test_processed = processed_results["test"]

    for df_name, df in {
        "train_processed": train_processed,
        "val_processed": val_processed,
        "test_processed": test_processed,
    }.items():
        if "hour" in df.columns:
            df.drop(columns=["hour"], inplace=True)
            logging.info(f"Dropped 'hour' column from {df_name}")

    logging.info("🧩 Train columns:")
    for col in train_processed.columns:
        logging.info(f"  - {col}")

    logging.info("🧩 Val columns:")
    for col in val_processed.columns:
        logging.info(f"  - {col}")

    if preprocess_only:
        logging.info("🛑 preprocess_only=True → Skipping training and exiting.")
        return

    logging.info("▶ Building feature matrices...")
    t0 = time.time()
    train_features, val_features, feature_cols, cat_features = build_feature_matrix(
        train_processed,
        val_processed
    )
    logging.info(f"✓ Feature matrix built in {(time.time() - t0)/60:.2f} min")
    logging.info(f"  ▶ Categorical features: {cat_features}")

    _, test_features, _, _ = build_feature_matrix(train_processed, test_processed)

    for c in cat_features:
        logging.info(f"  - {c} dtype (train): {train_features[c].dtype}, (val): {val_features[c].dtype}")

    if "pv_id" in feature_cols:
        feature_cols = [c for c in feature_cols if c != "pv_id"]

    if "pv_id" in cat_features:
        cat_features = [c for c in cat_features if c != "pv_id"]

    X_train = train_features[feature_cols]
    y_train = train_processed[TARGET].clip(lower=0).values
    X_val = val_features[feature_cols]
    y_val = val_processed[TARGET].clip(lower=0).values
    y_val_raw = val_processed[TARGET].values

    val_preds = train_single_model(gpu, X_train, X_val, y_train, y_val, cat_features, cfg["model_dir"])

    mae = mean_absolute_error(y_val_raw, val_preds)
    logging.info(f"🏁 Final Validation MAE = {mae:.5f}")
    logging.info(f"✅ All done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
