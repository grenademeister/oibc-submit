# OIBC Submission Pipeline

This repository contains the complete pipeline for the OIBC submission, including data processing, clustering, feature engineering, model training, and inference.

## Repository Structure

```
.
├── run_pipeline.py          # Orchestration script for the complete pipeline
├── demo.ipynb               # Demonstration Jupyter notebook
├── scripts/            # scripts for clustering and data splitting
│   ├── add_cluster.py
│   ├── add_cluster_efficient.py
│   └── add_cluster_from_train.py
│   └── split.py
└── src/                     # Main training and inference code
    ├── config.yaml          # Configuration file
    ├── main.py              # Ensemble training
    ├── main_lgbm.py         # LightGBM training
    ├── main_catboost.py     # CatBoost training
    ├── main_xgb.py          # XGBoost training
    ├── preprocess.py        # Feature engineering
    ├── weight.py            # Grid search ensemble
    ├── weight_fixed.py      # Fixed weight ensemble
    └── compare_features.py  # Feature comparison utility
```

## Quick Start

### 1. Using the Orchestration Script

Run the complete pipeline with a single command:

```bash
python run_pipeline.py --config src/config.yaml
```

### 2. Using the Jupyter Notebook

For interactive exploration and visualization:

```bash
jupyter notebook demo.ipynb
```

## Pipeline Steps

The pipeline consists of 5 main steps:

### 1. Data Splitting
Split the raw data into training and validation sets based on unique PV IDs.

```bash
python run_pipeline.py --step split
```

### 2. Clustering
Create location-based clusters for PV systems using K-Means clustering.

```bash
python run_pipeline.py --step cluster
```

### 3. Feature Engineering
Generate cluster-based features (integrated into training step).

```bash
python run_pipeline.py --step features
```

### 4. Model Training
Train ensemble or individual models (LightGBM, CatBoost, XGBoost).

```bash
python run_pipeline.py --step train
```

### 5. Inference
Generate predictions using trained models.

```bash
python run_pipeline.py --step infer
```

## Configuration

Edit `src/config.yaml` to set: model type (`mode`), GPU usage (`use_gpu`), cluster count (`n_clusters`), and data paths.

## Advanced Usage

```bash
# Skip preprocessing steps
python run_pipeline.py --skip-split --skip-cluster

# Train individual models
cd src && python main_lgbm.py    # or main_catboost.py, main_xgb.py

# Ensemble inference
cd src && python weight.py         # grid search weights
cd src && python weight_fixed.py   # fixed weights
```

## Requirements

Install dependencies:
```bash
pip install pandas numpy scikit-learn lightgbm catboost xgboost joblib pyyaml pvlib-python pytz jupyter matplotlib seaborn
```
