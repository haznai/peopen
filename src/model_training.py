# imports
from model_definition import (
    HyperParams,
    Dataset,
    LanguageModel,
    Logger,
    Metrics,
    PenPrompterNetwork,
    Trainer,
    get_train_and_valid_path,
)

from submodels.factual_consistency_model import (
    FactualConsistencyNetwork,
)

from dspy import BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPROv2
import dspy

# Loading in the data
train_path, valid_path = get_train_and_valid_path()

# Initialize components that will be reused
_ = Logger()  # just needs to be init
data = Dataset(train_pickle_path=str(train_path), valid_pickle_path=str(valid_path))
metrics = Metrics()
fc_model = FactualConsistencyNetwork()
fc_model.load(
    "models/factual_consistency/2024-11-01_bootsrapfewshotwithrandomsearch_second.json"
)
fc_model._compiled = True
network = PenPrompterNetwork(fc_model)

# Define training configurations
training_configs = {
    "boostrapfewshot": {
        "optimizer": BootstrapFewShot(metric=metrics.get_score, max_rounds=4),
        "lm": dspy.LM("openai/gpt-4o-mini"),
    },
    "bootsrapfewshotwithrandomsearch": {
        "optimizer": BootstrapFewShotWithRandomSearch(
            metric=metrics.get_score,
            num_threads=1,
            num_candidate_programs=4,
            max_rounds=4,
        ),
        "lm": dspy.LM("openai/gpt-4o-mini"),
    },
    "miprov2fewshot": {
        "optimizer": MIPROv2(metric=metrics.get_score, num_threads=1, num_candidates=4),
        "lm": dspy.LM("openai/gpt-4o-mini"),
    },
}


def run_all_training_configurations():
    for run_name, config in training_configs.items():
        print(f"\n{'='*50}")
        print(f"Starting training run: {run_name}")
        print(f"{'='*50}")

        try:
            # Create new HyperParams for this run
            current_params = HyperParams(
                training_run_name=f"penprompter-{run_name}",
                fc_weights_path="models/factual_consistency/2024-11-01_bootsrapfewshotwithrandomsearch_second.json",
            )

            # Initialize language model
            _ = LanguageModel(lm=config["lm"])

            # Create new Trainer instance
            trainer = Trainer(
                params=current_params,
                network=network,
                data=data,
                optimizer=config["optimizer"],
            )

            # Train the model
            trainer.optimize()

            print(f"\nSuccessfully completed training run: {run_name}")

        except Exception as e:
            print(f"\nError in training run {run_name}: {str(e)}")
            continue


if __name__ == "__main__":
    run_all_training_configurations()
