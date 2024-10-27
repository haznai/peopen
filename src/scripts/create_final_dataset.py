# %% imports
import pickle
import pandas as pd

from dspy.datasets import DataLoader

# %% load datasets that aren't in the dspy.example format
with open("../../data/processed/merged_dataset_train.pkl", "rb") as f:
    merged_train = pickle.load(f)

with open("../../data/processed/merged_dataset_valid.pkl", "rb") as f:
    merged_valid = pickle.load(f)


# %% convert dicts to pandas format
df_merged_train = pd.DataFrame(merged_train)
df_merged_valid = pd.DataFrame(merged_valid)

# List of columns to keep
columns_to_keep = [
    "titel",
    "argumenteBundesrat",
    "empfehlungBundesrat",
    "empfehlungKomitee",
    "im_detail",
    "argumenteKomitee",
    "wortlaut_filled",
]

# Process training dataframe
df_merged_train = df_merged_train[columns_to_keep]  # Keep only specified columns
df_merged_train = df_merged_train.rename(columns={"wortlaut_filled": "Wortlaut"})

# Process validation dataframe
df_merged_valid = df_merged_valid[columns_to_keep]  # Keep only specified columns
df_merged_valid = df_merged_valid.rename(columns={"wortlaut_filled": "Wortlaut"})


train_dataset = DataLoader().from_pandas(
    df_merged_train,
    input_keys=tuple([col for col in df_merged_train.columns if col != "Wortlaut"]),
)

valid_dataset = DataLoader().from_pandas(
    df_merged_valid,
    input_keys=tuple([col for col in df_merged_valid.columns if col != "Wortlaut"]),
)

# %% save datasets as pickles
with open("../../data/processed/final_truncated_train_dataset.pkl", "wb") as f:
    pickle.dump(train_dataset, f)

with open("../../data/processed/final_truncated_valid_dataset.pkl", "wb") as f:
    pickle.dump(valid_dataset, f)
