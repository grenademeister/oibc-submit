# OIBC Solar Irradiance Prediction Pipeline

Machine learning pipeline for predicting solar irradiance (nins) using gradient boosting models with spatial clustering and temporal feature engineering.

## Pipeline Architecture

### 1. Data Splitting

**Script:** `data_split/split.py`

Splits training data by photovoltaic installation ID to prevent data leakage.

- Splits on unique `pv_id` values (not individual rows)
- Default validation ratio: 10%
- Maintains temporal sequences within each PV installation

### 2. Spatial Clustering

**Scripts:**
- `cluster_code/add_cluster_efficient.py` - Train set clustering
- `closest_code/add_cluster_from_train.py` - Test/validation set clustering

Performs K-means clustering on PV installation locations to capture spatial patterns.

**Process:**
- Clusters PV installations by geographic coordinates (coord1, coord2)
- Computes per-cluster hourly statistics (uv_idx, nins, humidity)
- Generates distance-based ratios for all clusters: `1 / (1 + distance)^2`
- For test/validation sets: identifies K nearest clusters and applies training cluster statistics

**Variants:**
- `p34/`: Uses all-cluster features (features from all K clusters)
- `p34_closest/`: Uses top-5 closest clusters only

### 3. Feature Engineering

**Script:** `preprocess.py`

Generates temporal, solar position, and weather interaction features.

**Feature Categories:**

Temporal:
- Cyclical encodings (sin/cos) for time-of-day and day-of-year
- Day of week, month, season indicators

Solar:
- Solar altitude and azimuth using pvlib
- Solar altitude normalized to [0,1]
- Binary sun-up indicator

Weather Interactions:
- Temperature range (max - min)
- Surface temperature differential (temp_a - temp_b)
- Pressure gap (atmospheric - ground)
- Wind resultant magnitude and gust ratio
- Cloud cover mean
- Dewpoint spread
- Precipitation binary indicator

Missing values handled via forward/backward fill within PV groups.

### 4. Model Training

**Script:** `main.py`

Trains gradient boosting models with optional GPU acceleration.

**Supported Models:**
- LightGBM
- CatBoost
- XGBoost

**Training Modes:**
- Single model: Trains one specified model
- Ensemble: Trains all three models and performs grid search over ensemble weights

**Configuration:** `config.yaml`

Key parameters:
- Model selection via `mode` parameter
- GPU acceleration toggle
- Paths to clustered feature sets
- Model hyperparameters in `BASE_MODEL_PARAMS`

**Ensemble Weight Selection:**

Grid search over weight combinations (0.0 to 1.0 in 0.1 increments) using validation MAE as optimization criterion.

### 5. Model Variants

**Directory Structure:**
- `main_lgbm.py` - LightGBM-specific implementation
- `main_catboost.py` - CatBoost-specific implementation
- `main_xgb.py` - XGBoost-specific implementation
- `main_multi.py` - Multi-model ensemble implementation
- `weight.py` / `weight_fixed.py` - Ensemble weight optimization utilities

## Execution Sequence

```
1. data_split/split.py
   → train_split.csv, val_split.csv

2. cluster_code/add_cluster_efficient.py
   → train_cluster_model.joblib
   → train_with_cluster_means_and_ratios.joblib

3. closest_code/add_cluster_from_train.py
   → val_with_cluster_means_and_ratios.joblib
   → test_with_cluster_means_and_ratios.joblib

4. p34/main.py (or p34_closest/main.py)
   → trained models in model_dir
   → validation predictions
   → ensemble weights (if applicable)
```

## Configuration

Primary configuration file: `config.yaml`

Required paths:
- `train_cluster_features`: Clustered training features (joblib)
- `val_cluster_features`: Clustered validation features (joblib)
- `test_cluster_features`: Clustered test features (joblib)
- `save_path`: Output directory for models and logs

Model configuration in `BASE_MODEL_PARAMS` dictionary within `main.py`.

## Dependencies

Core:
- pandas, numpy
- scikit-learn
- joblib

Models:
- lightgbm
- catboost
- xgboost

Feature engineering:
- pvlib
- pytz

## Output

Training produces:
- Trained model files (pickle format)
- Validation predictions
- Ensemble weights (if ensemble mode)
- Training logs with MAE metrics
- Model configuration snapshots
