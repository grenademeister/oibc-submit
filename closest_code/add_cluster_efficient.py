#!/usr/bin/env python3
import pandas as pd
from sklearn.cluster import KMeans
import joblib
import os
import numpy as np

# ==========================================
# 0️⃣ CONFIGURATION
# ==========================================
data_path = "/workspace/oibc/data_split/train_split.csv"
num_cluster = 12
num_closest = 5     # <<<< you control this
save_prefix = "train"

# Columns to compute per-cluster means
select_column = ["uv_idx", "nins", "humidity"]

# ==========================================
# 1️⃣ LOAD DATA
# ==========================================
df = pd.read_csv(data_path, parse_dates=['time'])
print(f"✅ Loaded dataset with shape {df.shape}")

# Preserve original row order for verification later
df["_orig_idx"] = np.arange(len(df))
orig_idx_snapshot = df["_orig_idx"].copy()

# ==========================================
# 2️⃣ CLEAN COORDINATES
# ==========================================
if not {'coord1', 'coord2'}.issubset(df.columns):
    raise ValueError("Columns 'coord1' and 'coord2' are required for clustering.")

pv_location = (
    df[['pv_id', 'coord1', 'coord2']]
    .dropna(subset=['coord1', 'coord2'])
    .drop_duplicates(subset=['pv_id'], keep='first')
)

# ==========================================
# 3️⃣ CLUSTER PVs BY LOCATION
# ==========================================
kmeans = KMeans(n_clusters=num_cluster, random_state=42)
pv_location['cluster'] = kmeans.fit_predict(pv_location[['coord1', 'coord2']]).astype(int)

# Save model + centroids (efficient binary format)
joblib.dump(kmeans, f"{save_prefix}_cluster_model.joblib", compress=3)
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['coord1', 'coord2'])
joblib.dump(centroids, f"{save_prefix}_cluster_centroids.joblib", compress=3)

# Merge cluster labels
df = df.merge(pv_location[['pv_id', 'cluster']], on='pv_id', how='left')

# ==========================================
# 4️⃣ ADD HOUR COLUMN AND COMPUTE PER-HOUR CLUSTER MEANS
# ==========================================
df['hour'] = df['time'].dt.floor('H')

cluster_means = (
    df.groupby(['hour', 'cluster'])[select_column]
      .mean()
      .reset_index()
)

# ==========================================
# 5️⃣ FIND TOP-K CLOSEST CLUSTERS
# ==========================================
coords = df[['coord1', 'coord2']].to_numpy()
centers = kmeans.cluster_centers_

# Distance matrix N×K
dists = np.sqrt(((coords[:, None, :] - centers[None, :, :])**2).sum(axis=2))

# Sorted clusters by distance
closest_idx = np.argsort(dists, axis=1)
closest_k = closest_idx[:, :num_closest]

# Create cluster index columns
for i in range(num_closest):
    df[f"closest{i+1}_cluster"] = closest_k[:, i]

# ==========================================
# 6️⃣ LOOKUP MEANS FOR ONLY CLOSEST CLUSTERS
# ==========================================
# Build (hour, cluster) → dict({col: value})
lookup = {}
for row in cluster_means.itertuples(index=False):
    lookup[(row.hour, row.cluster)] = {col: getattr(row, col) for col in select_column}

# Add mean columns for closest clusters
for i in range(num_closest):
    cl_col = f"closest{i+1}_cluster"
    for col in select_column:
        out_col = f"closest{i+1}_{col}"
        df[out_col] = [
            lookup.get((h, c), {}).get(col, np.nan)
            for h, c in zip(df['hour'], df[cl_col])
        ]

# ==========================================
# 7️⃣ ADD DISTANCE-BASED RATIOS FOR CLOSEST CLUSTERS
# ==========================================
coords = df[['coord1', 'coord2']].to_numpy()

for i in range(num_closest):
    cl_col = f"closest{i+1}_cluster"
    ratio_col = f"closest{i+1}_ratio"

    ratios = []
    for (x, y), cidx in zip(coords, df[cl_col]):
        cx, cy = kmeans.cluster_centers_[cidx]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        ratios.append(1 / (1 + dist)**2)

    df[ratio_col] = ratios

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

# Drop helper index before saving
df = df.drop(columns="_orig_idx")

# ==========================================
# 9️⃣ SAVE RESULTS (JOBLIB ONLY)
# ==========================================
output_path = f"{save_prefix}_closest_with_ratios.joblib"
joblib.dump(df, output_path, compress=3)

print("✅ Extended clustering + top-K cluster means + ratios pipeline complete!")
print(f"📦 Output saved to: {os.path.abspath(output_path)}")
print(f"🧭 Final shape: {df.shape}")
print(f"Added features: {[f'closest{i+1}_{col}' for i in range(num_closest) for col in select_column]}")
print(f"Ratio features: {[f'closest{i+1}_ratio' for i in range(num_closest)]}")
