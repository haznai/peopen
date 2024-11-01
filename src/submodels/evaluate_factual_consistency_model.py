from typing import List
import dspy
import sys
import numpy as np
from factual_consistency_model import (
    FactualConsistencyNetwork,
    FactualConsistencyMetrics,
    load_model,
)
import pickle

sys.path.append("/Users/hazn/Desktop/code.nosync/peopen")

from src.model_definition import get_train_and_valid_path, LanguageModel


def load_validation_data() -> List[dspy.Example]:
    """Load the validation dataset."""
    _, valid_path = get_train_and_valid_path()

    with open(valid_path, "rb") as f:
        valid_dataset = pickle.load(f)
        if not valid_dataset:
            raise ValueError("Validation dataset is empty or wrongly loaded")

    return valid_dataset


def evaluate_factual_consistency():
    """
    Evaluate factual consistency on the validation set.

    Args:
        sample_size: Optional number of examples to evaluate. If None, evaluates all examples.
    """
    # Initialize models
    _ = LanguageModel()
    fc_transformer = load_model()
    metrics = FactualConsistencyMetrics(model=fc_transformer)

    # Initialize and compile network
    fc_network = FactualConsistencyNetwork()
    fc_network.load(
        "../../models/factual_consistency/2024-10-31_boostrapfewshot_second.json"
    )
    fc_network._compiled = True

    # Load validation data
    validation_data = load_validation_data()

    scores = []
    total = len(validation_data)

    # Evaluate each example
    for idx, example in enumerate(validation_data):
        print(f"Processing {idx + 1}/{total}", end="\r")

        try:
            # Get predictions
            prediction = fc_network(
                titel=example.titel,
                im_detail=example.im_detail,
                argumenteKomitee=example.argumenteKomitee,
                empfehlungKomitee=example.empfehlungKomitee,
                argumenteBundesrat=example.argumenteBundesrat,
                empfehlungBundesrat=example.empfehlungBundesrat,
            )

            # Calculate score
            score = metrics.get_score(example, prediction)
            scores.append(score)

        except Exception as e:
            print(f"\nError processing example {idx}: {str(e)}")

    # Calculate and print statistics
    valid_scores = [s for s in scores if s is not None]

    print("\nEvaluation Results:")
    print("-" * 20)
    print(f"Total examples evaluated: {total}")
    print(f"Successfully processed: {len(valid_scores)}")
    print(f"Failed processing: {total - len(valid_scores)}")

    if valid_scores:
        print("\nScore Statistics:")
        print(f"Mean score: {np.mean(valid_scores):.4f}")
        print(f"Median score: {np.median(valid_scores):.4f}")
        print(f"Std dev: {np.std(valid_scores):.4f}")
        print(f"Min score: {min(valid_scores):.4f}")
        print(f"Max score: {max(valid_scores):.4f}")


evaluate_factual_consistency()
