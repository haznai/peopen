# %% Imports
from article_number_retrieval_model import (
    get_path_to_truncated_wortlaut_pickles,
    ArticleNumberRM,
)
import pickle
import pandas as pd
from tabulate import tabulate  # Added this import
from datetime import datetime

from dspy.datasets import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
)

from typing import TypeAlias

# %% Loading and preprocessing the data
train_path, valid_path = get_path_to_truncated_wortlaut_pickles()

with open(train_path, "rb") as f:
    data_train = pickle.load(f)


with open(valid_path, "rb") as f:
    data_valid = pickle.load(f)

print(data_train[0])


# Create df_train
df_train = pd.DataFrame(data_train)

# Create df_valid
df_valid = pd.DataFrame(data_valid)

# Now you can use these DataFrames to create your datasets
train_dataset = DataLoader().from_pandas(
    df_train,
    input_keys=tuple([col for col in df_train.columns if col != "wortlaut_original"]),
)

valid_dataset = DataLoader().from_pandas(
    df_valid,
    input_keys=tuple([col for col in df_valid.columns if col != "wortlaut_original"]),
)

# %% functions for evaluating the model
ListOfArticleNumbers: TypeAlias = list[list[str | None]]


def fill_out_the_preds_vs_targets(
    all_preds: list[list[str]], all_targets: list[list[str]]
) -> tuple[ListOfArticleNumbers, ListOfArticleNumbers]:
    """
    Fill out the preds and targets so that they have the same length.
    Because of the LLM nature of the Retrieval model, we don't always get the same
    number of predicted article numbers as the target article numbers.

    This function fills out the shorter list with `None` values.
    """
    filled_preds = []
    filled_targets = []

    for preds, targets in zip(all_preds, all_targets):
        max_length = max(len(preds), len(targets))
        filled_preds.append(preds + [None] * (max_length - len(preds)))
        filled_targets.append(targets + [None] * (max_length - len(targets)))

    return (filled_preds, filled_targets)


def eval(
    retrieved_preds: list[list[str | None]], retrieved_targets: list[list[str | None]]
) -> dict[str, float | str]:
    """
    Evaluate multilabel classification performance for article number retrieval.

    Args:
        retrieved_preds: List of lists containing predicted article numbers (or None)
        retrieved_targets: List of lists containing target article numbers (or None)

    Returns:
        Dictionary containing evaluation metrics and timestamp
    """
    # Collect all valid labels
    all_labels = {
        label
        for labels in retrieved_preds + retrieved_targets
        for label in labels
        if label is not None
    }

    if not all_labels:
        raise ValueError("No valid labels found in predictions or targets")

    # Create label mapping and binary vectors
    label_to_index = {label: idx for idx, label in enumerate(sorted(all_labels))}

    def binarize(labels: list[str | None]) -> list[int]:
        binary = [0] * len(label_to_index)
        for label in labels:
            if label is not None and label in label_to_index:
                binary[label_to_index[label]] = 1
        return binary

    y_true = [binarize(labels) for labels in retrieved_targets]
    y_pred = [binarize(labels) for labels in retrieved_preds]

    # Calculate metrics with explicit zero_division handling
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics = {
        "Model": current_time,
        "Hamming Score (Accuracy)": 1 - hamming_loss(y_true, y_pred),
        "Exact Match Ratio (Subset Accuracy)": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "Precision Score": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "Recall Score": recall_score(y_true, y_pred, average="micro", zero_division=0),
    }

    # Add timestamp

    # Create row and headers for display
    headers = list(metrics.keys())
    row = [
        f"{metrics[h]:.4f}" if isinstance(metrics[h], float) else metrics[h]
        for h in headers
    ]

    # Display tables
    table_settings = {
        "headers": headers,
        "floatfmt": ".4f",
        "maxcolwidths": None,
        "tablefmt": None,
    }

    print("\nGitHub Format:")
    table_settings["tablefmt"] = "github"
    print(tabulate([row], **table_settings))

    table_settings["tablefmt"] = "latex"
    print("\nLaTeX Format:")
    print(tabulate([row], **table_settings))

    table_settings["tablefmt"] = "plain"
    print("\nCompact Format (horizontal scroll):")
    print(tabulate([row], **table_settings))

    return metrics


# %% Init and run the model
article_number_rm = ArticleNumberRM()

retrieved_preds = []
retrieved_targets = []

# Process training dataset
print("Processing training dataset...")
for i, row in enumerate(train_dataset):
    parsed_articles = article_number_rm.forward(
        query_or_queries=row["wortlaut_truncated"]
    )
    retrieved_preds.append(parsed_articles.retrieved_article_numbers)
    retrieved_targets.append(row["article_numbers"])
    print(f"Processed training example {i + 1}/{len(train_dataset)}")

# Process validation dataset
print("\nProcessing validation dataset...")
for i, row in enumerate(valid_dataset):
    parsed_articles = article_number_rm.forward(
        query_or_queries=row["wortlaut_truncated"]
    )
    retrieved_preds.append(parsed_articles.retrieved_article_numbers)
    retrieved_targets.append(row["article_numbers"])
    print(f"Processed validation example {i + 1}/{len(valid_dataset)}")

# %% Call the eval function
retrieved_preds, retrieved_targets = fill_out_the_preds_vs_targets(
    retrieved_preds, retrieved_targets
)
eval(retrieved_preds, retrieved_targets)
