# custom imports
from model_definition import (
    PenPrompterNetwork,
    Metrics,
    Dataset,
    LanguageModel,
    get_train_and_valid_path,
    get_model_path,
)

# dspy
from dspy.evaluate.evaluate import Evaluate

# helper imports
import os


# %% I/O
model_path = get_model_path("2024-10-07_first_run_2024-10-06.json")
trained_network = PenPrompterNetwork()
trained_network.load(model_path)

train_path, valid_path = get_train_and_valid_path()
data = Dataset(train_pickle_path=str(train_path), valid_pickle_path=str(valid_path))
trainset = data.data["train"]
valset = data.data["validation"]
metrics = Metrics()
lm = LanguageModel(
    name="gpt-4o-mini-2024-07-18",
    url="https://api.openai.com/v1/",
    api_key=os.environ.get("OPENAI_API_KEY"),  # type: ignore
)


# %% Evaluation
# validation set
val_evaluate = Evaluate(
    devset=valset,
    metric=metrics.get_score,
    num_threads=1,
    display_table=False,
    display_progress=False,
    return_all_scores=True,
)


val_score, val_subscores = val_evaluate(trained_network, return_all_scores=True)  # type: ignore

# training set
train_evaluate = Evaluate(
    devset=trainset,
    metric=metrics.get_score,
    num_threads=1,
    display_table=False,
    display_progress=False,
    return_all_scores=True,
)
train_score, train_subscores = train_evaluate(trained_network, return_all_scores=True)  # type: ignore


# %% Print results
# pretty print the results in a table, like:
# | model_name                     |
# | metric | Validation | Training |
# |--------|------------|----------|
# | score  | 0.123      | 0.456    |
# | subscores   | [0.123, ...]      | [0.456, ....]   |

col_width = 15
total_width = col_width * 3 + 4  # 3 columns + 4 for separators

# Print the table
print(f"Model: {model_path.name}")
print("=" * total_width)
print(
    f"| {'Metric'.ljust(col_width)}| {'Validation'.center(col_width)}| {'Training'.center(col_width)}|"
)
print("|" + "-" * (total_width - 2) + "|")

# Print score row
print(
    f"| {'score'.ljust(col_width)}| {val_score:.4f}".ljust(col_width * 2 + 2)
    + f"| {train_score:.4f}".rjust(col_width)
    + " |"
)


# Print subscores row
def format_subscores(subscores):
    if not subscores:
        return "N/A"
    formatted = ", ".join([f"{s:.4f}" for s in subscores])
    return (
        (formatted[: col_width - 5] + "...")
        if len(formatted) > col_width - 2
        else formatted
    )


val_sub = format_subscores(val_subscores)
train_sub = format_subscores(train_subscores)
print(
    f"| {'subscores'.ljust(col_width)}| {val_sub.ljust(col_width)}| {train_sub.ljust(col_width)}|"
)
print("=" * total_width)
