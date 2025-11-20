import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Local variables
csv_path = "/home/user/gy/oibc/code_wrap/oibc_code/data/train.csv"   # path to your CSV file
train_path = "/home/user/gy/oibc/code_wrap/oibc_code/data_split/train_split.csv"
val_path = "/home/user/gy/oibc/code_wrap/oibc_code/data_split/val_split.csv"
seed = 42
ratio = 0.1

# Load dataset
data = pd.read_csv(csv_path)

# Get unique pv_ids
pv_ids = data["pv_id"].unique()

# Split pv_ids into train and validation sets
train_ids, val_ids = train_test_split(
    pv_ids,
    test_size=ratio,         # ratio is for validation set
    random_state=seed,
    shuffle=True
)

# Filter the original dataframe based on pv_ids
train_df = data[data["pv_id"].isin(train_ids)]
val_df = data[data["pv_id"].isin(val_ids)]

# Ensure output directory exists
os.makedirs(os.path.dirname(train_path), exist_ok=True)

# Save the splits
train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)

print(f"Training set: {len(train_df)} rows, {len(train_ids)} unique pv_ids")
print(f"Validation set: {len(val_df)} rows, {len(val_ids)} unique pv_ids")
print("Done.")
