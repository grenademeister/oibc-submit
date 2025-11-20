#!/usr/bin/env python3
import joblib
from pathlib import Path

# --------------------------------------------------------------------------------====
# CONFIG — use only the paths from your config.yaml
# --------------------------------------------------------------------------------====
train_cluster_features = Path(
    "/workspace/oibc/cluster/train_with_cluster_means_and_ratios.joblib"
)
val_cluster_features = Path(
    "/workspace/oibc/cluster/val_with_cluster_means_and_ratios.joblib"
)

# --------------------------------------------------------------------------------====
# LOAD DATA
# --------------------------------------------------------------------------------====
print("   Loading cluster feature files...")
train_df = joblib.load(train_cluster_features)
val_df = joblib.load(val_cluster_features)

print(f"   Train shape: {train_df.shape}")
print(f"   Val shape:   {val_df.shape}")

# --------------------------------------------------------------------------------====
# COMPARE FEATURE COLUMNS
# --------------------------------------------------------------------------------====
train_cols = set(train_df.columns)
val_cols = set(val_df.columns)

missing_in_val = sorted(train_cols - val_cols)
extra_in_val = sorted(val_cols - train_cols)
common_cols = sorted(train_cols & val_cols)

print("\n--------------------=== CLUSTER FEATURE COMPARISON --------------------===")
print(f"Train columns: {len(train_cols)}")
print(f"Val columns:   {len(val_cols)}")
print(f"Common:        {len(common_cols)}")
print("--------------------------------------------------------------")

if missing_in_val:
    print("  Warning: Missing in validation:")
    for col in missing_in_val:
        print(f"  - {col}")
else:
    print("   No missing columns in validation")

if extra_in_val:
    print("\n  Warning: Extra in validation:")
    for col in extra_in_val:
        print(f"  - {col}")
else:
    print("\n   No extra columns in validation")

print(
    "--------------------------------------------------------------------------------======\n"
)
