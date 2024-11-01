import sys

sys.path.append("/Users/hazn/Desktop/code.nosync/peopen")

from src.model_definition import (
    HyperParams,
    Dataset,
    LanguageModel,
    Logger,
    Trainer,
    get_train_and_valid_path,
)

from factual_consistency_model import (
    FactualConsistencyMetrics,
    FactualConsistencyNetwork,
    load_model,
)

import os
from dspy import BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPROv2

# Get data paths
train_path, valid_path = get_train_and_valid_path()

# Initialize common components that will be reused across all runs
lm = LanguageModel(
    name="gpt-4o-mini-2024-07-18",
    url="https://api.openai.com/v1/",
    api_key=os.environ.get("OPENAI_API_KEY"),  # type: ignore
)
_ = Logger()  # Initialize logger
data = Dataset(train_pickle_path=str(train_path), valid_pickle_path=str(valid_path))
metrics = FactualConsistencyMetrics(load_model())
network = FactualConsistencyNetwork()

# Define multiple training configurations
training_configs = {
    "boostrapfewshot_second": BootstrapFewShot(metric=metrics.get_score, max_rounds=4),
    "bootsrapfewshotwithrandomsearch_second": BootstrapFewShotWithRandomSearch(
        metric=metrics.get_score, num_threads=1, num_candidate_programs=4, max_rounds=4
    ),
    "miprov2fewshot_second": MIPROv2(
        metric=metrics.get_score, num_threads=1, num_candidates=4
    ),
}


# Run training for each configuration
def run_all_training_configurations():
    for run_name, optimizer in training_configs.items():
        print(f"\n{'='*50}")
        print(f"Starting training run: {run_name}")
        print(f"{'='*50}")

        try:
            # Create new HyperParams for this run
            current_params = HyperParams(training_run_name=run_name)

            # Create new Trainer instance
            trainer = Trainer(
                params=current_params, network=network, data=data, optimizer=optimizer
            )

            # Train the model
            trainer.optimize()

            print(f"\nSuccessfully completed training run: {run_name}")

        except Exception as e:
            print(f"\nError in training run {run_name}: {str(e)}")
            continue


run_all_training_configurations()
