#!/usr/bin/env python3
import logging, warnings, yaml, pickle, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from lightgbm import LGBMRegressor
import lightgbm
from itertools import product
from sklearn.metrics import mean_absolute_error


try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from preprocess import (
    load_dataset, create_spatial_clusters, apply_spatial_clusters,
    engineer_features, build_feature_matrix, pv_based_split, encode_for_xgb, TARGET
)

warnings.filterwarnings("ignore")

MODEL_NAMES = ("lightgbm", "catboost", "xgboost")

# ---------------------------------------------------------------------
# Model parameter definitions
# ---------------------------------------------------------------------
# BASE_MODEL_PARAMS: Dict[str, Dict] = {
#     "lightgbm": {
#         "objective": "regression",
#         "metric": "mae",
#         "n_estimators": 5000,
#         "learning_rate": 0.046349,
#         "num_leaves": 256,
#         "subsample": 0.9,
#         "colsample_bytree": 0.8,
#         "random_state": 42,
#         "n_jobs": -1,
#         "verbose": -1,
#     },
#     "catboost": {
#         "loss_function": "MAE",
#         "eval_metric": "MAE",
#         "iterations": 3000,
#         "learning_rate": 0.03,
#         "depth": 15,
#         "l2_leaf_reg": 5.0,
#         "random_seed": 42,
#         "allow_writing_files": False,
#     },
#     "xgboost": {
#         "objective": "reg:squarederror",
#         "n_estimators": 5000,
#         "learning_rate": 0.04,
#         "max_depth": 11,
#         "subsample": 0.856,
#         "colsample_bytree": 0.835,
#         "random_state": 42,
#         "n_jobs": -1,
#         "verbosity": 1,
#         "eval_metric": "mae",
#     },
# }



BASE_MODEL_PARAMS: Dict[str, Dict] = {
    "lightgbm": {
        "objective": "regression",
        "metric": "mae",
        "n_estimators": 3000,
        "learning_rate": 0.046349,
        "num_leaves": 110,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "catboost": {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 3000,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "allow_writing_files": False,
    },
    "xgboost": {
        "objective": "reg:squarederror",
        "n_estimators": 3000,
        "learning_rate": 0.05,
        "max_depth": 9,
        "subsample": 0.856,
        "colsample_bytree": 0.835,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 1,
        "eval_metric": "mae",
    },
}

GPU_MODEL_PARAMS: Dict[str, Dict] = {
    "lightgbm": {"device": "cuda", "gpu_use_dp": False},
    "catboost": {"task_type": "GPU", "devices": "0"},
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

def build_model(model_name: str, use_gpu=False):
    params = BASE_MODEL_PARAMS[model_name].copy()
    if use_gpu:
        params.update(GPU_MODEL_PARAMS.get(model_name, {}))
    if model_name == "lightgbm":
        return LGBMRegressor(**params)
    if model_name == "catboost":
        return CatBoostRegressor(**params)
    if model_name == "xgboost":
        return XGBRegressor(**params)
    raise ValueError(f"Unknown model {model_name}")

# ---------------------------------------------------------------------
# Parallel training function
# ---------------------------------------------------------------------
def train_single_model(model_name, gpu, xgb_data, X_train, X_val, X_test, y_train, y_val, cat_features, model_dir):
    start_time = time.time()
    logging.info(f"   START training {model_name.upper()}")

    model = build_model(model_name, use_gpu=gpu)

    if model_name == "xgboost":
        eval_set = [(xgb_data["xgb_val"], y_val)]
        model.fit(xgb_data["xgb_train"], y_train, eval_set=eval_set, verbose=100)
        val_preds = model.predict(xgb_data["xgb_val"])
        test_preds = model.predict(xgb_data["xgb_test"])

    elif model_name == "catboost":
        cat_idx = [list(X_train.columns).index(c) for c in cat_features]
        model.fit(X_train, y_train, cat_features=cat_idx, eval_set=(X_val, y_val),
                  use_best_model=False, verbose=100)
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)

    else:  # LightGBM
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="mae",
                  categorical_feature=cat_features or None,
                  callbacks=[lightgbm.log_evaluation(period=100)])
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)

    val_preds = np.expm1(val_preds).clip(0)
    test_preds = np.expm1(test_preds).clip(0)

    model_path = Path(model_dir) / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logging.info(f"   FINISHED {model_name.upper()} ({time.time() - start_time:.2f}s)")
    return model_name, val_preds, test_preds

# ---------------------------------------------------------------------
# Grid search for best ensemble weights
# ---------------------------------------------------------------------
def find_best_weights(val_preds_list, y_val_raw, top_k: int = 4):
    """Grid search ensemble weights and report top K combinations by MAE."""
    n_models = len(val_preds_list)
    grid = np.linspace(0, 1, 11)  # coarse 0.0 → 1.0, step 0.1
    results = []

    for weights in product(grid, repeat=n_models):
        weights = np.array(weights)
        if np.isclose(weights.sum(), 0):
            continue
        weights /= weights.sum()
        ensemble_pred = sum(w * p for w, p in zip(weights, val_preds_list))
        score = mean_absolute_error(y_val_raw, ensemble_pred)
        results.append((score, weights))

    results.sort(key=lambda x: x[0])
    best_score, best_weights = results[0]
    logging.info(f"   Best MAE on validation = {best_score:.5f} with weights {best_weights}")

    logging.info("📊 Top 4 weight combinations:")
    for i, (score, weights) in enumerate(results[:top_k], 1):
        logging.info(f"  {i}. MAE={score:.5f}, Weights={weights}")

    return best_weights

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    cfg = load_config("config.yaml")
    log_path = Path(cfg["setting_dir"]) / "training.log"
    setup_logging(log_path)

    # 🆕 Save base model parameters for reproducibility
    params_file = Path(cfg["setting_dir"]) / "base_model_params.yaml"
    with open(params_file, "w") as f:
        yaml.safe_dump(BASE_MODEL_PARAMS, f)
    logging.info(f"   Saved base model parameters → {params_file}")

    gpu = check_gpu_available() and cfg["use_gpu"]
    if cfg["use_gpu"] and not gpu:
        logging.warning("GPU requested but not available — running on CPU.")

    nrows = cfg["num_subset"] if cfg.get("use_subset", False) else None
    train = load_dataset(Path(cfg["data_path"]) / cfg["train_file"], nrows=nrows)
    test = load_dataset(Path(cfg["data_path"]) / cfg["test_file"], nrows=nrows)

    logging.info("   Feature processing...")
    t0 = time.time()
    train = create_spatial_clusters(train, cfg["n_clusters"], save_dir=cfg["model_dir"])
    test = apply_spatial_clusters(test, load_dir=cfg["model_dir"])
    train_processed = engineer_features(train).dropna(subset=[TARGET])
    test_processed = engineer_features(test)
    train_features, test_features, feature_cols, cat_features = build_feature_matrix(train_processed, test_processed)
    logging.info(f"   Feature engineering done in {(time.time() - t0)/60:.2f} min")

    modeling_frame = train_processed[["time", TARGET]].join(train_features)
    
    X_train, X_val, X_full, y_train, y_val, y_val_raw, y_full = pv_based_split(modeling_frame, feature_cols, val_fraction=0.04, random_state=42)
    
    X_test = test_features[feature_cols].copy()
    xgb_data = {k: encode_for_xgb(v, cat_features)
                for k, v in zip(["xgb_train", "xgb_val", "xgb_full", "xgb_test"],
                                [X_train, X_val, X_full, X_test])}

    # Train models in parallel
    model_list = MODEL_NAMES if cfg["mode"] == "ensemble" else [cfg["mode"]]
    val_preds_list, test_preds_list = [], []

    with ProcessPoolExecutor(max_workers=len(model_list)) as executor:
        futures = {
            executor.submit(
                train_single_model, model_name, gpu,
                xgb_data, X_train, X_val, X_test,
                y_train, y_val, cat_features, cfg["model_dir"]
            ): model_name for model_name in model_list
        }
        for future in as_completed(futures):
            _, val_preds, test_preds = future.result()
            val_preds_list.append(val_preds)
            test_preds_list.append(test_preds)

    # Find best ensemble weights using validation data
    if len(model_list) > 1:
        weights = find_best_weights(val_preds_list, y_val_raw)
        ensemble_preds = sum(w * p for w, p in zip(weights, test_preds_list))
        logging.info(f"   Using optimized weights: {weights}")
    else:
        ensemble_preds = test_preds_list[0]
        logging.info(f"   Single model mode: {model_list[0]}")

    # Save predictions
    submission = load_dataset(Path(cfg["data_path"]) / cfg["submission_file"])
    if len(ensemble_preds) != len(submission):
        submission = submission.iloc[:len(ensemble_preds)].copy()
    submission["nins"] = ensemble_preds
    output_csv = Path(cfg["save_path"]) / "predictions.csv"
    submission.to_csv(output_csv, index=False)
    logging.info(f"   Saved predictions to {output_csv}")
    logging.info(f"   All done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
