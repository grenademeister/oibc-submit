#!/usr/bin/env python3
"""
Orchestration script for the OIBC submission pipeline.

This script runs the complete workflow:
1. Data splitting (train/validation)
2. Clustering (PV location-based clustering)
3. Feature engineering (cluster features for train/val/test)
4. Model training (ensemble or single model)
5. Inference (generate predictions)

Usage:
    python run_pipeline.py --config src/config.yaml [--step STEP]

    Steps: split, cluster, features, train, infer, all (default)
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
import yaml


def setup_logging():
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_config(config_path):
    """Load configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_command(cmd, description):
    """Run a shell command and log the output."""
    logging.info(f"Starting: {description}")
    logging.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            logging.info(result.stdout)
        logging.info(f"Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed: {description}")
        logging.error(f"Error: {e.stderr}")
        return False


def step_split(config):
    """Step 1: Split data into train/validation sets."""
    logging.info("=" * 60)
    logging.info("STEP 1: Data Splitting")
    logging.info("=" * 60)

    return run_command(["python", "scripts/split.py"], "Data splitting")


def step_cluster(config):
    """Step 2: Create clusters based on PV locations."""
    logging.info("=" * 60)
    logging.info("STEP 2: Clustering")
    logging.info("=" * 60)

    # Run clustering on training data
    if not run_command(
        ["python", "scripts/add_cluster_efficient.py"], "Clustering training data"
    ):
        return False

    # Add cluster features to validation data
    if not run_command(
        ["python", "scripts/add_cluster_from_train.py"],
        "Adding cluster features to validation data",
    ):
        return False

    return True


def step_features(config):
    """Step 3: Engineer features for all datasets."""
    logging.info("=" * 60)
    logging.info("STEP 3: Feature Engineering")
    logging.info("=" * 60)

    # This step is typically integrated into the training script
    # via preprocess.py which is imported by main.py
    logging.info("Feature engineering will be performed during training")
    return True


def step_train(config):
    """Step 4: Train models."""
    logging.info("=" * 60)
    logging.info("STEP 4: Model Training")
    logging.info("=" * 60)

    mode = config.get("mode", "ensemble")

    if mode == "ensemble":
        script = "src/main.py"
    elif mode == "lightgbm":
        script = "src/main_lgbm.py"
    elif mode == "catboost":
        script = "src/main_catboost.py"
    elif mode == "xgboost":
        script = "src/main_xgb.py"
    else:
        script = "src/main.py"

    return run_command(["python", script], f"Model training ({mode})")


def step_infer(config):
    """Step 5: Generate predictions."""
    logging.info("=" * 60)
    logging.info("STEP 5: Inference")
    logging.info("=" * 60)

    # Use weight_fixed.py for inference with fixed ensemble weights
    return run_command(["python", "src/weight_fixed.py"], "Generating predictions")


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate the OIBC submission pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config.yaml",
        help="Path to configuration file (default: src/config.yaml)",
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["split", "cluster", "features", "train", "infer", "all"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip data splitting (if already done)",
    )
    parser.add_argument(
        "--skip-cluster", action="store_true", help="Skip clustering (if already done)"
    )

    args = parser.parse_args()

    setup_logging()

    logging.info("=" * 60)
    logging.info("OIBC Submission Pipeline")
    logging.info("=" * 60)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    logging.info(f"Loaded configuration from: {config_path}")
    logging.info(f"Mode: {config.get('mode', 'ensemble')}")
    logging.info(f"GPU: {config.get('use_gpu', False)}")

    # Define pipeline steps
    steps = {
        "split": step_split,
        "cluster": step_cluster,
        "features": step_features,
        "train": step_train,
        "infer": step_infer,
    }

    # Determine which steps to run
    if args.step == "all":
        step_sequence = ["split", "cluster", "features", "train", "infer"]
        if args.skip_split:
            step_sequence.remove("split")
        if args.skip_cluster:
            step_sequence.remove("cluster")
    else:
        step_sequence = [args.step]

    # Execute steps
    success = True
    for step_name in step_sequence:
        if not steps[step_name](config):
            logging.error(f"Pipeline failed at step: {step_name}")
            success = False
            break

    if success:
        logging.info("=" * 60)
        logging.info("Pipeline completed successfully!")
        logging.info("=" * 60)

        # Display output paths
        save_path = Path(config.get("save_path", "./output"))
        if save_path.exists():
            logging.info(f"\nOutput directory: {save_path}")
            model_dir = save_path / "model"
            if model_dir.exists():
                models = list(model_dir.glob("*.pkl"))
                logging.info(f"Models saved: {len(models)}")

            predictions = list(save_path.glob("predictions_*.csv"))
            if predictions:
                logging.info(f"\nPredictions generated:")
                for pred_file in predictions:
                    logging.info(f"  - {pred_file.name}")
    else:
        logging.error("=" * 60)
        logging.error("Pipeline failed!")
        logging.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
