from typing import List
from pathlib import Path
import dspy
import sys
import numpy as np
from tabulate import tabulate
from factual_consistency_model import (
    FactualConsistencyNetwork,
    FactualConsistencyMetrics,
    load_model,
)
import pickle

sys.path.append("/Users/hazn/Desktop/code.nosync/peopen")
from src.model_definition import get_train_and_valid_path, LanguageModel


def get_model_weights() -> List[Path]:
    """Get all json weight files from the factual consistency models directory."""
    weights_dir = Path("../../models/factual_consistency")
    return list(weights_dir.glob("*.json"))


def load_validation_data() -> List[dspy.Example]:
    """Load the validation dataset."""
    _, valid_path = get_train_and_valid_path()
    with open(valid_path, "rb") as f:
        valid_dataset = pickle.load(f)
        if not valid_dataset:
            raise ValueError("Validation dataset is empty or wrongly loaded")
    return valid_dataset


def evaluate_model(
    network: FactualConsistencyNetwork,
    metrics: FactualConsistencyMetrics,
    data: List[dspy.Example],
) -> float:
    """Evaluate a single model and return its mean score."""
    scores = []
    total = len(data)

    for idx, example in enumerate(data):
        print(f"Processing {idx + 1}/{total}", end="\r")
        try:
            prediction = network(
                titel=example.titel,
                im_detail=example.im_detail,
                argumenteKomitee=example.argumenteKomitee,
                empfehlungKomitee=example.empfehlungKomitee,
                argumenteBundesrat=example.argumenteBundesrat,
                empfehlungBundesrat=example.empfehlungBundesrat,
            )
            score = metrics.get_score(example, prediction)
            if score is not None:  # Only append valid scores
                scores.append(score)
        except Exception as e:
            print(f"\nError processing example {idx}: {str(e)}")

    return np.mean(scores) if scores else 0.0


def evaluate_all_models():
    """Evaluate factual consistency across all model weights."""
    # Initialize
    _ = LanguageModel()
    metrics = FactualConsistencyMetrics(model=load_model())
    validation_data = load_validation_data()

    # Evaluate each model
    results = []
    for weight_file in get_model_weights():
        print(f"\nEvaluating model: {weight_file.name}")

        network = FactualConsistencyNetwork()
        network.load(str(weight_file))
        network._compiled = True

        mean_score = evaluate_model(network, metrics, validation_data)
        results.append(["gpt4o", weight_file.name, f"{mean_score:.4f}"])

    # Print results in different formats
    headers = ["Model", "Weights", "Mean Score"]
    for format in ["github", "latex", "simple"]:
        print(f"\n{format.capitalize()} Format:")
        print(tabulate(results, headers=headers, tablefmt=format))


if __name__ == "__main__":
    evaluate_all_models()
