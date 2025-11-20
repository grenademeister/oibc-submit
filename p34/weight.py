#!/usr/bin/env python3
"""
Weighted ensemble inference (fixed weights by model name)
Safe for LightGBM, XGBoost, and CatBoost.
DEBUG VERSION
"""

import logging, warnings, yaml, pickle, joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

from preprocess import (
    build_feature_matrix,
    encode_for_xgb,
    TARGET,
)

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
def setup_logging(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, "w"), logging.StreamHandler()],
    )

# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------
def main():
    # --------------------------------------------------------------
    # Load config & set up logging
    # --------------------------------------------------------------
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    save_path = Path(cfg["save_path"])
    model_dir = save_path / "model"
    log_path = save_path / "inference_weighted_fixed_debug.log"
    setup_logging(log_path)

    logging.info("   Starting weighted-fixed inference (DEBUG mode, LightGBM first)...")

    processed_dir = (
        Path(cfg["processed_cache_path"])
        if "processed_cache_path" in cfg and cfg["processed_cache_path"]
        else save_path / "processed_features"
    )
    logging.info(f"   Using processed features from: {processed_dir}")

    # --------------------------------------------------------------
    # Load processed frames
    # --------------------------------------------------------------
    train = joblib.load(processed_dir / "train_processed.joblib")
    val = joblib.load(processed_dir / "val_processed.joblib")
    test = joblib.load(processed_dir / "test_processed.joblib")
    logging.info(f"   Loaded: train {train.shape}, val {val.shape}, test {test.shape}")

    # Debug initial dtypes
    for name, df in [("train", train), ("val", val), ("test", test)]:
        pv_dtype = df["pv_id"].dtype if "pv_id" in df.columns else "MISSING"
        cl_dtype = df["cluster"].dtype if "cluster" in df.columns else "MISSING"
        # DEBUG: logging.info(f" [{name}] BEFORE cast → pv_id: {pv_dtype}, cluster: {cl_dtype}")

    # --------------------------------------------------------------
    # Ensure pv_id / cluster are categorical BEFORE feature matrix
    # --------------------------------------------------------------
    for df in (train, val, test):
        if "pv_id" in df.columns:
            df["pv_id"] = df["pv_id"].astype("category")
        if "cluster" in df.columns:
            df["cluster"] = df["cluster"].astype("category")

    for name, df in [("train", train), ("val", val), ("test", test)]:
        pv_dtype = df["pv_id"].dtype if "pv_id" in df.columns else "MISSING"
        cl_dtype = df["cluster"].dtype if "cluster" in df.columns else "MISSING"
        # DEBUG: logging.info(f" [{name}] AFTER cast → pv_id: {pv_dtype}, cluster: {cl_dtype}")

    # --------------------------------------------------------------
    # Drop 'hour' if present
    # --------------------------------------------------------------
    for df_name, df in {"train": train, "val": val, "test": test}.items():
        if "hour" in df.columns:
            df.drop(columns=["hour"], inplace=True)
            logging.info(f"Dropped 'hour' column from {df_name}")

    # --------------------------------------------------------------
    # Build feature matrices (same logic as training)
    # --------------------------------------------------------------
    logging.info("   Building feature matrices (train/val)...")
    train_features, val_features, feature_cols, cat_features = build_feature_matrix(train, val)
    logging.info(f"  → feature_cols: {len(feature_cols)}")
    logging.info(f"  → cat_features: {cat_features}")

    logging.info("   Building feature matrices (train/test)...")
    _, test_features, _, _ = build_feature_matrix(train, test)

    # Debug dtypes after feature matrix
    for c in cat_features:
        logging.info(
            f"DEBUG after build_feature_matrix: "
            f"train[{c}]={train_features[c].dtype}, val[{c}]={val_features[c].dtype}, "
            f"test[{c}]={test_features[c].dtype}"
        )
    
    if "pv_id" in feature_cols:
        feature_cols = [c for c in feature_cols if c != "pv_id"]

    if "pv_id" in cat_features:
        cat_features = [c for c in cat_features if c != "pv_id"]

    # --------------------------------------------------------------
    # Final matrices for models
    # --------------------------------------------------------------
    X_train = train_features[feature_cols]
    X_val = val_features[feature_cols]
    X_test = test_features[feature_cols]

    # DEBUG: logging.info(f": X_train shape = {X_train.shape}")
    # DEBUG: logging.info(f": X_val shape   = {X_val.shape}")
    # DEBUG: logging.info(f": X_test shape  = {X_test.shape}")

    y_val = val[TARGET].clip(lower=0).values
    y_val_raw = val[TARGET].values

    xgb_data = {
        k: encode_for_xgb(v.copy(), cat_features)
        for k, v in zip(["xgb_train", "xgb_val", "xgb_test"], [X_train, X_val, X_test])
    }
    logging.info("DEBUG: Encoded data for XGBoost")

    # --------------------------------------------------------------
    # Load models (LightGBM first)
    # --------------------------------------------------------------
    model_files = sorted([p for p in model_dir.glob("*.pkl") if p.name != "cluster_map.pkl"])
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")

    def model_sort_key(p: Path):
        name = p.stem.lower()
        if "lightgbm" in name:
            return 0
        if "xgb" in name:
            return 1
        if "catboost" in name:
            return 2
        return 3

    model_files = sorted(model_files, key=model_sort_key)
    model_names = [m.stem for m in model_files]
    logging.info(f"Found models (sorted): {model_names}")

    val_preds_list, test_preds_list = [], []
    data_path = Path(cfg["data_path"])
    submission_path = data_path / cfg.get("submission_file", "submission.csv")

    # --------------------------------------------------------------
    # Predict per model
    # --------------------------------------------------------------
    for model_file in model_files:
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        name = model_file.stem
        lname = name.lower()

        logging.info("--------------------")
        logging.info(f"   Predicting with model: {name}")
        logging.info("--------------------")

        # fresh copies
        Xv = X_val.copy()
        Xt = X_test.copy()


        # ------------------ XGBoost ------------------
        if "xgb" in lname:
            logging.info("DEBUG: XGBoost branch")

            # 🔥 NEW DEBUG
            # DEBUG: logging.info(f" Xv.columns = {list(X_val.columns)}")
            # DEBUG: logging.info(f" feature_cols = {feature_cols}")
            # DEBUG: logging.info(f" cat_features = {cat_features}")
            # DEBUG: logging.info(f" xgb_val columns = {list(xgb_data['xgb_val'].columns)}")

            val_preds = model.predict(xgb_data["xgb_val"])
            test_preds = model.predict(xgb_data["xgb_test"])


        # ------------------ LightGBM ------------------
        elif "lightgbm" in lname:
            logging.info("DEBUG: LightGBM branch")

            # Use model's feature order if available
            model_feature_names = getattr(model, "feature_name_", list(X_train.columns))
            # DEBUG: logging.info(f": LightGBM expects {len(model_feature_names)} features")

            # Reindex to model's expected columns (order + subset)
            Xv = Xv.reindex(columns=model_feature_names)
            Xt = Xt.reindex(columns=model_feature_names)

            val_preds = model.predict(Xv)
            test_preds = model.predict(Xt)

        

        # ------------------ CatBoost ------------------
        elif "catboost" in lname:
            logging.info("DEBUG: CatBoost branch")
            # CatBoost can handle numeric + categorical internally if trained that way
            val_preds = model.predict(Xv)
            test_preds = model.predict(Xt)

        # ------------------ Fallback ------------------
        else:
            logging.info("DEBUG: Fallback model branch (plain predict)")
            val_preds = model.predict(Xv)
            test_preds = model.predict(Xt)

        # Non-negative clip
        val_preds = np.clip(val_preds, 0, None)
        test_preds = np.clip(test_preds, 0, None)

        mae = mean_absolute_error(y_val_raw, val_preds)
        logging.info(f"  → Validation MAE for {name}: {mae:.6f}")

        val_preds_list.append(val_preds)
        test_preds_list.append(test_preds)

        # Save individual model predictions
        out_individual = save_path / f"predictions_{name}.csv"
        if submission_path.exists():
            submission_model = pd.read_csv(submission_path)
            target_col = "nins" if "nins" in submission_model.columns else submission_model.columns[-1]
            submission_model[target_col] = test_preds
        else:
            submission_model = pd.DataFrame({"nins": test_preds})
        submission_model.to_csv(out_individual, index=False)
        logging.info(f"   Saved individual predictions → {out_individual}")

        # --------------------------------------------------------------
    # GRID SEARCH: recursive simplex over all model weights
    # --------------------------------------------------------------
    logging.info("   Starting GRID SEARCH over all model weights")

    step = 0.1
    grid = np.arange(0.0, 1.0 + step, step)
    n_models = len(model_names)

    results = []   # will store tuples: (mae, weight_vector)

    # ------------------------------
    # Recursive weight generator
    # ------------------------------
    def dfs(level, current_weights):
        s = sum(current_weights)
        if s > 1.0:
            return

        # last model → remaining weight = 1 - sum
        if level == n_models - 1:
            w_last = 1.0 - s
            if w_last < 0 or w_last > 1:
                return

            w = current_weights + [w_last]
            w = np.array(w)

            blended = np.zeros_like(val_preds_list[0])
            for wi, pi in zip(w, val_preds_list):
                blended += wi * pi

            mae = mean_absolute_error(y_val_raw, blended)
            results.append((mae, w))
            return

        # otherwise loop
        for wv in grid:
            dfs(level + 1, current_weights + [wv])

    dfs(0, [])

    # sort best 10
    results.sort(key=lambda x: x[0])
    best_mae, best_w = results[0]

    logging.info(f"   BEST MAE = {best_mae:.6f}")
    logging.info("   BEST WEIGHTS:")
    for name, w in zip(model_names, best_w):
        logging.info(f"  {name}: {w:.3f}")

    logging.info("\n   TOP 10 Weight Combinations:")
    logging.info(f"Model order → {model_names}")
    for i, (mae, w) in enumerate(results[:10], 1):
        logging.info(f"{i}. MAE={mae:.6f}, weights={np.round(w, 3).tolist()}")

    # Build final ensemble with best weights
    ensemble_preds = np.zeros_like(test_preds_list[0])
    for wi, pi, name in zip(best_w, test_preds_list, model_names):
        logging.info(f"  → Using {name} weight={wi:.3f}")
        ensemble_preds += wi * pi

    # --------------------------------------------------------------
    # Save ensemble predictions
    # --------------------------------------------------------------
    out_csv = save_path / "predictions_weighted_grid.csv"
    if submission_path.exists():
        submission = pd.read_csv(submission_path)
        target_col = "nins" if "nins" in submission.columns else submission.columns[-1]
        submission[target_col] = ensemble_preds
    else:
        submission = pd.DataFrame({"nins": ensemble_preds})

    submission.to_csv(out_csv, index=False)
    logging.info(f"   Saved GRID-SEARCH ensemble predictions → {out_csv}")
    logging.info("   Grid search inference finished successfully.")



# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
