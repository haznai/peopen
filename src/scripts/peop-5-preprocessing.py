import pandas as pd
import json
import sqlite3

# Load in data
with sqlite3.connect("../../data/processed/volksiniativen_with_wortlaut.db") as conn:
    raw_df = pd.read_sql(
        'SELECT "Titel der Vorlage", "Daten aus der Broschüre", "Wortlaut", "Datum" FROM volksabstimmungen',
        conn,
    )


# Function to parse JSON and handle potential errors
def parse_json(json_str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}


# Parse the JSON strings
parsed_data = raw_df["Daten aus der Broschüre"].apply(parse_json)

# Create a new DataFrame from the parsed data
json_df = pd.DataFrame(parsed_data.tolist())

# Combine the original DataFrame with the new JSON-derived columns
result_df = pd.concat(
    [raw_df[["Titel der Vorlage", "Wortlaut", "Datum"]], json_df], axis=1
)

# Parse the dates of the "Datum" column to datetime objects using 'mixed' format
result_df["Datum"] = pd.to_datetime(result_df["Datum"], format="mixed")

# Sort the DataFrame by date in descending order
result_df = result_df.sort_values("Datum", ascending=False)

# Split the data into train and validation sets
valid_df = result_df.head(4)  # Most recent 4 entries for validation
train_df = result_df.iloc[4:24]  # Next 20 entries for training

# Remove the date column from both train_df and valid_df
train_df = train_df.drop("Datum", axis=1)
valid_df = valid_df.drop("Datum", axis=1)

# Remove the Title der Vorlage column from both train_df and valid_df
train_df = train_df.drop("Titel der Vorlage", axis=1)
valid_df = valid_df.drop("Titel der Vorlage", axis=1)

# Print shapes to confirm
print("Train DataFrame shape:", train_df.shape)
print("Validation DataFrame shape:", valid_df.shape)

# Optional: Display the first few rows of each DataFrame
print("\nTrain DataFrame preview:")
train_df.head(1)
print("\nValidation DataFrame preview:")
valid_df.head(1)


def flatten_column(df, column_name):
    def flatten_list(row):
        if isinstance(row, list):
            return "\n\n".join(
                [
                    f"{item['ueberschrift']}\n{item['inhalt']}"
                    for item in row
                    if isinstance(item, dict)
                ]
            )
        return row

    df[column_name] = df[column_name].apply(flatten_list)
    return df


# Flatten the nested columns
columns_to_flatten = ["im_detail", "argumenteKomitee", "argumenteBundesrat"]
for column in columns_to_flatten:
    train_df = flatten_column(train_df, column)
    valid_df = flatten_column(valid_df, column)

# Print shapes to confirm
print("Train DataFrame shape after flattening:", train_df.shape)
print("Validation DataFrame shape after flattening:", valid_df.shape)

# Optional: Display the first few rows of each DataFrame to verify the changes
print("\nTrain DataFrame preview after flattening:")
print(train_df[columns_to_flatten].head(1))
print("\nValidation DataFrame preview after flattening:")
print(valid_df[columns_to_flatten].head(1))


from dspy.datasets import DataLoader

train_dataset = DataLoader().from_pandas(
    train_df, input_keys=tuple([col for col in train_df.columns if col != "Wortlaut"])
)

valid_dataset = DataLoader().from_pandas(
    valid_df, input_keys=tuple([col for col in valid_df.columns if col != "Wortlaut"])
)

train_dataset[0]
import pickle

# Save train dataset
with open(
    "../../data/processed/volksiniativen_with_wortlaut_dspy_dataset_train.pkl", "wb"
) as f:
    pickle.dump(train_dataset, f)

# Save valid dataset
with open(
    "../../data/processed/volksiniativen_with_wortlaut_dspy_dataset_valid.pkl", "wb"
) as f:
    pickle.dump(valid_dataset, f)

print("Datasets saved as pickles.")
# Load the saved datasets
with open(
    "../../data/processed/volksiniativen_with_wortlaut_dspy_dataset_train.pkl", "rb"
) as f:
    loaded_train_dataset = pickle.load(f)

with open(
    "../../data/processed/volksiniativen_with_wortlaut_dspy_dataset_valid.pkl", "rb"
) as f:
    loaded_valid_dataset = pickle.load(f)

# Check if the datasets are identical
train_identical = all(
    train_dataset[i] == loaded_train_dataset[i] for i in range(len(train_dataset))
)
valid_identical = all(
    valid_dataset[i] == loaded_valid_dataset[i] for i in range(len(valid_dataset))
)

print("\nSanity check results:")
print("Train datasets identical:", train_identical)
print("Validation datasets identical:", valid_identical)
