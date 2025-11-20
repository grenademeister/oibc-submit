# OIBC Submission Pipeline

This repository contains the complete pipeline for the OIBC (Open Innovation Big Competition) submission, including data processing, clustering, feature engineering, model training, and inference.

## Repository Structure

```
.
├── run_pipeline.py          # Orchestration script for the complete pipeline
├── demo.ipynb               # Demonstration Jupyter notebook
├── cluster_code/            # Clustering scripts
│   ├── add_cluster.py
│   ├── add_cluster_efficient.py
│   └── add_cluster_from_train.py
├── data_split/              # Data splitting utilities
│   └── split.py
└── p34/                     # Main training and inference code
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
python run_pipeline.py --config p34/config.yaml
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

Edit `p34/config.yaml` to customize:

- **mode**: Model type (`ensemble`, `lightgbm`, `catboost`, or `xgboost`)
- **use_gpu**: Enable GPU acceleration
- **n_clusters**: Number of clusters for K-Means
- **Data paths**: Input/output directories

Example configuration:

```yaml
mode: "xgboost"
use_gpu: true
n_clusters: 10
save_path: /workspace/oibc/data/result/exp34_all_1
data_path: /workspace/oibc/data
```

## Advanced Usage

### Run Specific Steps

```bash
# Skip data preparation steps if already done
python run_pipeline.py --skip-split --skip-cluster

# Run only training
python run_pipeline.py --step train

# Run only inference
python run_pipeline.py --step infer
```

### Train Individual Models

```bash
# Train LightGBM only
cd p34 && python main_lgbm.py

# Train CatBoost only
cd p34 && python main_catboost.py

# Train XGBoost only
cd p34 && python main_xgb.py
```

### Ensemble Methods

Two ensemble approaches are available:

1. **Grid Search Ensemble** (`weight.py`): Searches for optimal model weights
2. **Fixed Weight Ensemble** (`weight_fixed.py`): Uses predefined weights

```bash
# Grid search for best weights
cd p34 && python weight.py

# Use fixed weights
cd p34 && python weight_fixed.py
```

## Output

After running the pipeline, outputs are saved to the directory specified in `config.yaml`:

```
<save_path>/
├── model/                    # Trained model files
│   ├── lightgbm.pkl
│   ├── catboost.pkl
│   └── xgboost.pkl
├── setting/                  # Training configurations and logs
│   └── <timestamp>/
│       ├── config.yaml
│       ├── training.log
│       └── base_model_params.yaml
└── predictions_*.csv         # Prediction files
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- lightgbm
- catboost
- xgboost
- joblib
- pyyaml
- pvlib
- pytz

Install dependencies:

```bash
pip install pandas numpy scikit-learn lightgbm catboost xgboost joblib pyyaml pvlib-python pytz
```

For Jupyter notebook:

```bash
pip install jupyter matplotlib seaborn
```

## Tips

1. **GPU Acceleration**: Set `use_gpu: true` in config for faster training (requires GPU-enabled packages)
2. **Memory Management**: Use `use_subset: true` for testing with smaller datasets
3. **Logging**: Check logs in `<save_path>/setting/<timestamp>/training.log` for detailed training information
4. **Debugging**: Run individual scripts directly for easier debugging

## License

This project is for the OIBC competition submission.
