import pandas as pd
from sklearn.cluster import KMeans
import joblib
import os
import numpy as np

# ==========================================
# 0️⃣ CONFIGURATION
# ==========================================
data_path = "/home/user/gy/oibc/code_wrap/oibc_code/data_split/train_split.csv"
num_cluster = 10
save_prefix = "train"

# Columns to compute per-cluster means
select_column = ["uv_idx", "nins", "humidity"]

# ==========================================
# 1️⃣ LOAD DATA
# ==========================================
df = pd.read_csv(data_path, parse_dates=['time'])
print(f"✅ Loaded dataset with shape {df.shape}")

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

# Save model + centroids
joblib.dump(kmeans, f"{save_prefix}_cluster_model.joblib")
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['coord1', 'coord2'])
centroids.to_csv(f"{save_prefix}_cluster_centroids.csv", index_label='cluster_id')

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
# 5️⃣ BUILD WIDE TABLE (ALL CLUSTERS’ MEANS PER HOUR)
# ==========================================
df_cluster_all = None
for c in range(num_cluster):
    sub = cluster_means[cluster_means['cluster'] == c].copy()
    sub = sub.drop(columns='cluster')
    sub = sub.add_prefix(f'cluster_{c}_')
    sub = sub.rename(columns={f'cluster_{c}_hour': 'hour'})
    df_cluster_all = sub if df_cluster_all is None else pd.merge(df_cluster_all, sub, on='hour', how='outer')

# Merge this wide hourly table back
df = df.merge(df_cluster_all, on='hour', how='left')

# ==========================================
# 6️⃣ ADD DISTANCE-BASED RATIOS FOR ALL CLUSTERS
# ==========================================
# Compute distance to each centroid
coords = df[['coord1', 'coord2']].to_numpy()
for i, (cx, cy) in enumerate(kmeans.cluster_centers_):
    dist = np.sqrt((coords[:, 0] - cx)**2 + (coords[:, 1] - cy)**2)
    df[f'cluster_{i}_ratio'] = 1 / (1 + dist)**2

# ==========================================
# 7️⃣ OPTIONAL: FILL NaN VALUES (FORWARD + BACKWARD)
# ==========================================
df = (
    df.sort_values(['cluster', 'hour'])
      .groupby('cluster', group_keys=False)
      .apply(lambda g: g.ffill().bfill())
      .reset_index(drop=True)
)

# ==========================================
# 8️⃣ SAVE RESULTS
# ==========================================
output_path = f"{save_prefix}_with_cluster_means_and_ratios.csv"
df.to_csv(output_path, index=False)

print("✅ Extended clustering + all-cluster mean/ratio pipeline complete!")
print(f"📁 Output saved to: {os.path.abspath(output_path)}")
print(f"🧭 Final shape: {df.shape}")
print(f"Added per cluster: {[f'cluster_{i}_nins / cluster_{i}_ratio' for i in range(num_cluster)]}")
