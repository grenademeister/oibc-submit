#!/usr/bin/env python3
import pandas as pd
import joblib
import numpy as np
import os
CONFIGURATION

val_path = "/workspace/oibc/data_split/val_split.csv"
train_features_path = "train_with_cluster_means_and_ratios.joblib"
train_model_path = "train_cluster_model.joblib"
save_prefix = "val"

# Columns used when building per-cluster means in training
select_column = ["uv_idx", "nins", "humidity"]
LOAD VALIDATION DATA

df = pd.read_csv(val_path, parse_dates=["time"])
print(f"  Loaded validation data with shape {df.shape}")

# Preserve original row order for safety check
df["_orig_idx"] = np.arange(len(df))
orig_idx_snapshot = df["_orig_idx"].copy()
LOAD TRAINED CLUSTER MODEL + TRAINING CLUSTER MEANS

kmeans = joblib.load(train_model_path)
train_df = joblib.load(train_features_path)
num_cluster = len(kmeans.cluster_centers_)
print(f"  Loaded trained KMeans model with {num_cluster} clusters")

# Extract per-hour, per-cluster means from the training dataset
cluster_means = (
    train_df.groupby(["hour", "cluster"])[select_column]
    .mean()
    .reset_index()
)
print(f"  Extracted training cluster means: {cluster_means.shape}")
ASSIGN CLUSTERS TO VALIDATION PVs

if not {"coord1", "coord2"}.issubset(df.columns):
    raise ValueError("Columns 'coord1' and 'coord2' are required for clustering.")

pv_location = (
    df[["pv_id", "coord1", "coord2"]]
    .dropna(subset=["coord1", "coord2"])
    .drop_duplicates(subset=["pv_id"], keep="first")
)

pv_location["cluster"] = kmeans.predict(pv_location[["coord1", "coord2"]]).astype(int)
df = df.merge(pv_location[["pv_id", "cluster"]], on="pv_id", how="left")
print("  Assigned clusters to validation PVs using training centroids")
ADD HOUR COLUMN

df["hour"] = df["time"].dt.floor("H")
BUILD WIDE TABLE (ALL CLUSTERS’ MEANS PER HOUR)

df_cluster_all = None
for c in range(num_cluster):
    sub = cluster_means[cluster_means["cluster"] == c].copy()
    sub = sub.drop(columns="cluster")
    sub = sub.add_prefix(f"cluster_{c}_")
    sub = sub.rename(columns={f"cluster_{c}_hour": "hour"})
    df_cluster_all = sub if df_cluster_all is None else pd.merge(df_cluster_all, sub, on="hour", how="outer")

# Merge this wide hourly table back
df = df.merge(df_cluster_all, on="hour", how="left")
print(f"  Merged full cluster-wide hourly means into validation data ({df_cluster_all.shape[1]} extra columns)")
ADD DISTANCE-BASED RATIOS FOR ALL CLUSTERS

coords = df[["coord1", "coord2"]].to_numpy()
for i, (cx, cy) in enumerate(kmeans.cluster_centers_):
    dist = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
    df[f"cluster_{i}_ratio"] = 1 / (1 + dist) ** 2
print(f"  Added distance-based ratio features for all {num_cluster} clusters")
FILL NaN VALUES (FORWARD + BACKWARD, PRESERVE ORDER)

df = (
    df.sort_values(["cluster", "pv_id", "hour"])
      .groupby("cluster", group_keys=False)
      .apply(lambda g: g.ffill().bfill())
)

# Restore original CSV order
df = df.sort_values("_orig_idx").reset_index(drop=True)
VERIFY ROW ORDER PRESERVATION

if df["_orig_idx"].equals(orig_idx_snapshot):
    print("  Row order preserved correctly after fill")
else:
    print("⚠️ Row order changed after fill!")

# Drop helper column before saving
df = df.drop(columns="_orig_idx")
SAVE FINAL OUTPUT

output_path = f"{save_prefix}_with_cluster_means_and_ratios.joblib"
joblib.dump(df, output_path, compress=3)

print("\n  Validation dataset successfully built!")
print(f"  Saved to: {os.path.abspath(output_path)}")
print(f"  Final shape: {df.shape}")
