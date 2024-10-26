# %% imports
import pickle
from evaluate import load

# Load bertscore
bertscore = load("bertscore")


def fill_article_numbers(truncated_text: str, article_numbers: list[str]) -> str:
    """Replace [ARTICLE NUMBER HERE] placeholders with actual article numbers."""
    filled_text = truncated_text
    placeholder = "[ARTICLE NUMBER HERE]"

    # Assert we have the right number of article numbers for the placeholders
    placeholder_count = truncated_text.count(placeholder)
    assert (
        len(article_numbers) == placeholder_count
    ), f"Mismatch: {len(article_numbers)} numbers for {placeholder_count} placeholders"

    # Replace each placeholder with the corresponding article number
    for number in article_numbers:
        filled_text = filled_text.replace(placeholder, number, 1)

    return filled_text


def merge_datasets(
    truncated_dataset: list[dict], full_dataset: list[dict]
) -> list[dict]:
    """
    Merge datasets by matching wortlaut_original to Wortlaut using BERTScore and
    fill in article numbers in truncated text.
    """
    merged = []
    for trunc_item in truncated_dataset:
        # Compute similarity scores
        scores = bertscore.compute(
            predictions=[trunc_item["wortlaut_original"]] * len(full_dataset),
            references=[item["Wortlaut"] for item in full_dataset],
            model_type="bert-base-multilingual-cased",
            lang=["de"],
        )

        # Find best match
        best_match_idx = scores["f1"].index(max(scores["f1"]))
        full_item = full_dataset[best_match_idx]

        # Fill in article numbers and create merged item
        filled_wortlaut = fill_article_numbers(
            trunc_item["wortlaut_truncated"], trunc_item["article_numbers"]
        )

        merged_item = {
            **full_item,
            "wortlaut_truncated": trunc_item["wortlaut_truncated"],
            "wortlaut_filled": filled_wortlaut,
            "article_numbers": trunc_item["article_numbers"],
        }
        merged.append(merged_item)

    print(f"Matched {len(merged)} items")
    return merged


# %% load and merge
with open("../../data/processed/truncated_wortlaut_train.pkl", "rb") as f:
    truncated_wortlaut_train = pickle.load(f)

with open("../../data/processed/truncated_wortlaut_valid.pkl", "rb") as f:
    truncated_wortlaut_valid = pickle.load(f)

with open(
    "../../data/processed/volksiniativen_with_wortlaut_dspy_dataset_train.pkl", "rb"
) as f:
    full_dataset_train = pickle.load(f)

with open(
    "../../data/processed/volksiniativen_with_wortlaut_dspy_dataset_valid.pkl", "rb"
) as f:
    full_dataset_valid = pickle.load(f)

# Merge and save
merged_train = merge_datasets(truncated_wortlaut_train, full_dataset_train)
merged_valid = merge_datasets(truncated_wortlaut_valid, full_dataset_valid)

with open("../../data/processed/merged_dataset_train.pkl", "wb") as f:
    pickle.dump(merged_train, f)

with open("../../data/processed/merged_dataset_valid.pkl", "wb") as f:
    pickle.dump(merged_valid, f)

# %% check

with open("../../data/processed/merged_dataset_train.pkl", "rb") as f:
    merged_train = pickle.load(f)

with open("../../data/processed/merged_dataset_valid.pkl", "rb") as f:
    merged_valid = pickle.load(f)

print(merged_train[0])
print(merged_valid[0])
