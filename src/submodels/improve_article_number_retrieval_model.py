from article_number_retrieval_model import (
    get_path_to_truncated_wortlaut_pickles,
    ArticleNumberRM,
)
import pickle
import pandas as pd
from dspy.datasets import DataLoader
from tabulate import tabulate

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


def eval(all_preds: ListOfArticleNumbers, all_targets: ListOfArticleNumbers):
    """ """
    # Prepare data for the table
    for preds, targets in zip(all_preds, all_targets):
        table_data = []
        headers = ["Parsed Article", "Target Article", "Match"]
        for single_pred, single_target in zip(preds, targets):
            match = "✓" if single_pred == single_target else "✗"
            table_data.append([single_pred, single_target, match])
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # @todo: use roc-auc: https://huggingface.co/spaces/eval-metric/roc_auc
    # @todo: use accuracy


retrieved_preds, retrieved_targets = fill_out_the_preds_vs_targets(
    retrieved_preds, retrieved_targets
)
eval(retrieved_preds, retrieved_targets)
