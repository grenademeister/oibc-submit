#!/usr/bin/env python3
import pandas as pd
import joblib
import numpy as np
import os

# ==========================================
# 0️⃣ CONFIGURATION
# ==========================================
# val_path = "/workspace/oibc/data_split/val_split.csv"
# train_features_path = f"train_closest_with_ratios.joblib"   # ✅ FIXED
# train_model_path = "train_cluster_model.joblib"
# save_prefix = "val"

val_path = "/workspace/oibc/data/test.csv"
train_features_path = f"train_closest_with_ratios.joblib"   # ✅ FIXED
train_model_path = "train_cluster_model.joblib"
save_prefix = "test"


# how many closest clusters to keep
num_closest = 5

# Columns used when building per-cluster means in training
select_column = ["uv_idx", "nins", "humidity"]

# ==========================================
# 1️⃣ LOAD VALIDATION DATA
# ==========================================
df = pd.read_csv(val_path, parse_dates=["time"])
print(f"✅ Loaded validation data with shape {df.shape}")

# Preserve original row order for safety check
df["_orig_idx"] = np.arange(len(df))
orig_idx_snapshot = df["_orig_idx"].copy()

# ==========================================
# 2️⃣ LOAD TRAINED CLUSTER MODEL + TRAINING CLUSTER MEANS
# ==========================================
kmeans = joblib.load(train_model_path)
train_df = joblib.load(train_features_path)
num_cluster = len(kmeans.cluster_centers_)
print(f"✅ Loaded trained KMeans model with {num_cluster} clusters")

# Extract per-hour, per-cluster means from the training dataset
cluster_means = (
    train_df.groupby(["hour", "cluster"])[select_column]
    .mean()
    .reset_index()
)
print(f"✅ Extracted training cluster means: {cluster_means.shape}")

# ==========================================
# 3️⃣ ASSIGN CLUSTERS TO VALIDATION PVs
# ==========================================
if not {"coord1", "coord2"}.issubset(df.columns):
    raise ValueError("Columns 'coord1' and 'coord2' are required for clustering.")

pv_location = (
    df[["pv_id", "coord1", "coord2"]]
    .dropna(subset=["coord1", "coord2"])
    .drop_duplicates(subset=["pv_id"], keep="first")
)

pv_location["cluster"] = kmeans.predict(pv_location[["coord1", "coord2"]]).astype(int)
df = df.merge(pv_location[["pv_id", "cluster"]], on="pv_id", how="left")
print("✅ Assigned clusters to validation PVs using training centroids")

# ==========================================
# 4️⃣ ADD HOUR COLUMN
# ==========================================
df["hour"] = df["time"].dt.floor("H")

# ==========================================
# 5️⃣ FIND TOP-K CLOSEST CLUSTERS
# ==========================================
coords = df[["coord1", "coord2"]].to_numpy()
centers = kmeans.cluster_centers_

dists = np.sqrt(((coords[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
closest_idx = np.argsort(dists, axis=1)
closest_k = closest_idx[:, :num_closest]

for i in range(num_closest):
    df[f"closest{i+1}_cluster"] = closest_k[:, i]

# ==========================================
# 6️⃣ ADD MEANS FOR CLOSEST CLUSTERS (uv_idx, nins, humidity)
# ==========================================
lookup = {}
for row in cluster_means.itertuples(index=False):
    lookup[(row.hour, row.cluster)] = {col: getattr(row, col) for col in select_column}

for i in range(num_closest):
    ccol = f"closest{i+1}_cluster"
    for col in select_column:
        out = f"closest{i+1}_{col}"
        df[out] = [
            lookup.get((h, c), {}).get(col, np.nan)
            for h, c in zip(df["hour"], df[ccol])
        ]

coords = df[["coord1", "coord2"]].to_numpy()

for i in range(num_closest):
    ccol = f"closest{i+1}_cluster"
    rcol = f"closest{i+1}_ratio"

    ratios = []
    for (x, y), cid in zip(coords, df[ccol]):
        cx, cy = kmeans.cluster_centers_[cid]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        ratios.append(1 / (1 + dist)**2)

    df[rcol] = ratios



# ==========================================
# 8️⃣ FILL NaN VALUES (FORWARD + BACKWARD, PRESERVE ORDER)
# ==========================================
df = (
    df.sort_values(["cluster", "pv_id", "hour"])
      .groupby("cluster", group_keys=False)
      .apply(lambda g: g.ffill().bfill())
)

# Restore original CSV order
df = df.sort_values("_orig_idx").reset_index(drop=True)

# ==========================================
# ✅ VERIFY ROW ORDER PRESERVATION
# ==========================================
if df["_orig_idx"].equals(orig_idx_snapshot):
    print("✅ Row order preserved correctly after fill")
else:
    print("⚠️ Row order changed after fill!")

df = df.drop(columns="_orig_idx")

# ==========================================
# 9️⃣ SAVE FINAL OUTPUT
# ==========================================
output_path = f"{save_prefix}_closest_with_ratios.joblib"
joblib.dump(df, output_path, compress=3)

print("\n🎉 Validation dataset successfully built!")
print(f"📦 Saved to: {os.path.abspath(output_path)}")
print(f"🧭 Final shape: {df.shape}")
