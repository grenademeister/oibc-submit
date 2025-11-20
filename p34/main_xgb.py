#!/usr/bin/env python3
import logging, warnings, yaml, pickle, time, joblib
from datetime import datetime
from pathlib import Path
from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from preprocess import engineer_features, build_feature_matrix, encode_for_xgb, TARGET

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Model parameter definitions
# ---------------------------------------------------------------------
BASE_MODEL_PARAMS: Dict[str, Dict] = {
    "xgboost": {
        "objective": "reg:squarederror",
        "n_estimators": 6000,
        "learning_rate": 0.035,
        "max_depth": 11,
        "subsample": 0.856,
        "colsample_bytree": 0.835,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 1,
        "eval_metric": "mae",
    },
}

GPU_MODEL_PARAMS: Dict[str, Dict] = {
    "xgboost": {"device": "cuda", "tree_method": "hist"},
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
    params = BASE_MODEL_PARAMS["xgboost"].copy()
    if use_gpu:
        params.update(GPU_MODEL_PARAMS["xgboost"])
    return XGBRegressor(**params)

# ---------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------
def train_single_model(gpu, xgb_data, y_train, y_val, model_dir):
    start_time = time.time()
    logging.info("   START training XGBOOST")

    model = build_model(use_gpu=gpu)
    eval_set = [(xgb_data["xgb_val"], y_val)]
    model.fit(xgb_data["xgb_train"], y_train, eval_set=eval_set, verbose=100)
    val_preds = np.maximum(model.predict(xgb_data["xgb_val"]), 0)

    model_path = Path(model_dir) / "xgboost.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logging.info(f"   FINISHED XGBOOST ({time.time() - start_time:.2f}s)")
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

    #    Define processed cache directory (use from config if available)
    if "processed_cache_path" in cfg and cfg["processed_cache_path"]:
        processed_dir = Path(cfg["processed_cache_path"])
    else:
        processed_dir = Path(cfg["save_path"]) / "processed_features"

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train_processed.joblib"
    val_path = processed_dir / "val_processed.joblib"
    test_path = processed_dir / "test_processed.joblib"
    logging.info(f"Using processed cache path: {processed_dir}")

    #    Load pre-built cluster-enhanced feature sets
    logging.info("   Loading pre-built train/validation cluster features...")
    train = joblib.load(cfg["train_cluster_features"])
    val = joblib.load(cfg["val_cluster_features"])
    test = joblib.load(cfg["test_cluster_features"])
    logging.info(f"   Loaded train {train.shape}, val {val.shape}, test {test.shape}")

    # ### DEBUG: check pv_id presence in RAW
    for name, df in [("train_raw", train), ("val_raw", val), ("test_raw", test)]:
        has_pv = "pv_id" in df.columns
        logging.info(f"DEBUG [{name}] has pv_id column? {has_pv}")

    #    Use cached processed data if available (check each file individually)
    datasets = {
        "train": (train_path, train),
        "val": (val_path, val),
        "test": (test_path, test),
    }
    processed_results = {}

    for name, (path, raw_df) in datasets.items():
        if path.exists():
            logging.info(f"💾 Using cached {name}_processed from {path}")
            df_proc = joblib.load(path)
        else:
            logging.info(f"🧠 Processing {name} features (cache not found)...")
            df_proc = engineer_features(raw_df)
            if name == "train":
                df_proc = df_proc.dropna(subset=[TARGET])
            joblib.dump(df_proc, path)
            logging.info(f"💾 Saved {name}_processed to {path}")

        # ### DEBUG: pv_id in processed?
        logging.info(f"DEBUG [{name}_processed] has pv_id? {'pv_id' in df_proc.columns}")
        processed_results[name] = df_proc

    train_processed = processed_results["train"]
    val_processed = processed_results["val"]
    test_processed = processed_results["test"]

    # 🚫 Drop 'hour' column from processed data (safe for both cached/new)
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
    logging.info(f"DEBUG train_processed has pv_id? {'pv_id' in train_processed.columns}")

    logging.info("🧩 Val columns:")
    for col in val_processed.columns:
        logging.info(f"  - {col}")
    logging.info(f"DEBUG val_processed has pv_id? {'pv_id' in val_processed.columns}")

    #    If preprocess_only, exit after caching
    if preprocess_only:
        logging.info("🛑 preprocess_only=True → Skipping training and exiting.")
        return

    # Build aligned feature matrices
    logging.info("   Building feature matrices...")
    t0 = time.time()
    train_features, val_features, feature_cols, cat_features = build_feature_matrix(
        train_processed, val_processed
    )
    logging.info(f"   Feature matrix built in {(time.time() - t0)/60:.2f} min")

    # ### DEBUG: before any drop
    logging.info(f"DEBUG initial feature_cols length = {len(feature_cols)}")
    logging.info(f"DEBUG pv_id in feature_cols? {'pv_id' in feature_cols}")
    logging.info(f"DEBUG cat_features = {cat_features}")
    logging.info(f"DEBUG pv_id in cat_features? {'pv_id' in cat_features}")

    # ### pv_id FIX: remove from features & categoricals
    if "pv_id" in feature_cols:
        feature_cols = [c for c in feature_cols if c != "pv_id"]
        logging.info("⚠ Removed pv_id from feature_cols")
    if "pv_id" in cat_features:
        cat_features = [c for c in cat_features if c != "pv_id"]
        logging.info("⚠ Removed pv_id from cat_features")

    # ### DEBUG: after drop
    logging.info(f"DEBUG AFTER DROP: pv_id in feature_cols? {'pv_id' in feature_cols}")
    logging.info(f"DEBUG AFTER DROP: pv_id in cat_features? {'pv_id' in cat_features}")

    X_train = train_features[feature_cols]
    X_val = val_features[feature_cols]

    # ### DEBUG: check in matrices
    logging.info(f"DEBUG 'pv_id' in X_train.columns? {'pv_id' in X_train.columns}")
    logging.info(f"DEBUG 'pv_id' in X_val.columns?   {'pv_id' in X_val.columns}")

    # ⚠ IMPORTANT: labels should come from *processed* dfs, not raw train/val
    y_train = train_processed[TARGET].clip(lower=0).values
    y_val = val_processed[TARGET].clip(lower=0).values
    y_val_raw = val_processed[TARGET].values

    xgb_data = {
        k: encode_for_xgb(v, cat_features)
        for k, v in zip(["xgb_train", "xgb_val"], [X_train, X_val])
    }

    # ### DEBUG: type of xgb_data
    logging.info(f"DEBUG type(xgb_train) = {type(xgb_data['xgb_train'])}")
    logging.info(f"DEBUG type(xgb_val)   = {type(xgb_data['xgb_val'])}")

    # Train single XGBoost model
    val_preds = train_single_model(gpu, xgb_data, y_train, y_val, cfg["model_dir"])

    mae = mean_absolute_error(y_val_raw, val_preds)
    logging.info(f"   Final Validation MAE = {mae:.5f}")
    logging.info(f"   All done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
