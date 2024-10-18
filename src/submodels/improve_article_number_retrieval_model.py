from article_number_retrieval_model import (
    get_path_to_truncated_wortlaut_pickles,
    ArticleNumberRM,
)
import pickle
import pandas as pd
from dspy.datasets import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    roc_auc_score,
)

from typing import TypeAlias

# %% Loading and preprocessing the data
train_path, valid_path = get_path_to_truncated_wortlaut_pickles()

with open(train_path, "rb") as f:
    data_train = pickle.load(f)


with open(valid_path, "rb") as f:
    data_valid = pickle.load(f)


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

# %% Init the model
article_number_rm = ArticleNumberRM()

# %% get the article numbers
retrieved_preds = []
retrieved_targets = []

for i, row in enumerate(train_dataset):
    parsed_articles = article_number_rm.forward(
        query_or_queries=row["wortlaut_truncated"]
    )
    retrieved_preds.append(parsed_articles.retrieved_article_numbers)

    retrieved_targets.append(row["article_numbers"])

    print(f"Processed {i + 1}/{len(train_dataset)}")


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


def eval(retrieved_preds, retrieved_targets):
    """
    Evaluate multilabel classification performance.

    Parameters:
    - retrieved_preds: List of lists, each sublist contains predicted labels for an instance.
    - retrieved_targets: List of lists, each sublist contains true labels for an instance.
    """
    # Step 1: Collect all unique labels, excluding None
    all_labels = set()
    for labels in retrieved_preds + retrieved_targets:
        for label in labels:
            if label is not None:
                all_labels.add(label)

    # Step 2: Map labels to indices
    label_to_index = {label: idx for idx, label in enumerate(sorted(all_labels))}

    # Step 3: Function to binarize label lists, handling None values
    def binarize(labels):
        num_labels = len(label_to_index)
        binary_vector = [0] * num_labels
        for label in labels:
            if label is not None:
                idx = label_to_index[label]
                binary_vector[idx] = 1
            else:
                # If label is None, we skip it or handle it as needed
                continue
        return binary_vector

    # Step 4: Binarize the labels
    y_true = [binarize(labels) for labels in retrieved_targets]
    y_pred = [binarize(labels) for labels in retrieved_preds]

    # Step 5: Compute Exact Match Ratio (Subset Accuracy)
    subset_accuracy = accuracy_score(y_true, y_pred)
    print("Exact Match Ratio (Subset Accuracy): {:.4f}".format(subset_accuracy))

    # Step 6: Compute Hamming Loss and Hamming Score
    hamming_loss_value = hamming_loss(y_true, y_pred)
    hamming_score = 1 - hamming_loss_value
    print("Hamming Loss: {:.4f}".format(hamming_loss_value))
    print("Hamming Score (Accuracy): {:.4f}".format(hamming_score))

    # Step 7: Compute Precision, Recall, and F1 Score with micro averaging
    precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    print("Micro-Averaged Precision: {:.4f}".format(precision_micro))
    print("Micro-Averaged Recall: {:.4f}".format(recall_micro))
    print("Micro-Averaged F1 Score: {:.4f}".format(f1_micro))

    # Step 8: Compute Precision, Recall, and F1 Score with macro averaging
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("Macro-Averaged Precision: {:.4f}".format(precision_macro))
    print("Macro-Averaged Recall: {:.4f}".format(recall_macro))
    print("Macro-Averaged F1 Score: {:.4f}".format(f1_macro))


# Call the eval function
retrieved_preds, retrieved_targets = fill_out_the_preds_vs_targets(
    retrieved_preds, retrieved_targets
)
eval(retrieved_preds, retrieved_targets)

# %% print out retrieved_preds and retrieved_targets
print(f"retrieved_preds: {retrieved_preds}")
print(f"retrieved_targets: {retrieved_targets}")
