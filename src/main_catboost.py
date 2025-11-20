#!/usr/bin/env python3
import logging, warnings, yaml, pickle, time, joblib
from datetime import datetime
from pathlib import Path
from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

from preprocess import engineer_features, build_feature_matrix, encode_for_xgb, TARGET

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Model parameter definitions
# ---------------------------------------------------------------------
BASE_MODEL_PARAMS: Dict[str, Dict] = {
    "catboost": {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 2000,
        "learning_rate": 0.5,
        "depth": 11,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "allow_writing_files": False,
    },
}

GPU_MODEL_PARAMS: Dict[str, Dict] = {
    "catboost": {"task_type": "GPU", "devices": "0"},
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, "w"), logging.StreamHandler()],
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
    params = BASE_MODEL_PARAMS["catboost"].copy()
    if use_gpu:
        params.update(GPU_MODEL_PARAMS["catboost"])
    return CatBoostRegressor(**params)


# ---------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------
def train_single_model(gpu, cat_data, y_train, y_val, model_dir, cat_features):
    start_time = time.time()
    logging.info("   START training CATBOOST")

    model = build_model(use_gpu=gpu)
    model.fit(
        cat_data["cat_train"],
        y_train,
        eval_set=(cat_data["cat_val"], y_val),
        cat_features=cat_features,
        use_best_model=False,
        verbose=100,
    )

    val_preds = np.maximum(model.predict(cat_data["cat_val"]), 0)

    model_path = Path(model_dir) / "catboost.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logging.info(f"   FINISHED CATBOOST ({time.time() - start_time:.2f}s)")
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

    logging.info("   Loading pre-built train/validation cluster features...")
    train = joblib.load(cfg["train_cluster_features"])
    val = joblib.load(cfg["val_cluster_features"])
    test = joblib.load(cfg["test_cluster_features"])
    logging.info(f"   Loaded train {train.shape}, val {val.shape}, test {test.shape}")

    datasets = {
        "train": (train_path, train),
        "val": (val_path, val),
        "test": (test_path, test),
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

    logging.info("   Building feature matrices...")
    t0 = time.time()
    train_features, val_features, feature_cols, cat_features = build_feature_matrix(
        train_processed, val_processed
    )
    logging.info(f"   Feature matrix built in {(time.time() - t0)/60:.2f} min")

    X_train = train_features[feature_cols]
    y_train = train[TARGET].clip(lower=0).values
    X_val = val_features[feature_cols]
    y_val = val[TARGET].clip(lower=0).values
    y_val_raw = val[TARGET].values

    cat_data = {"cat_train": X_train, "cat_val": X_val}

    val_preds = train_single_model(
        gpu, cat_data, y_train, y_val, cfg["model_dir"], cat_features
    )

    mae = mean_absolute_error(y_val_raw, val_preds)
    logging.info(f"   Final Validation MAE = {mae:.5f}")
    logging.info(f"   All done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
