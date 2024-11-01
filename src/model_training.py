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
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

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

# Define language models to use
lm_list = ["gpt-4o-mini", "gpt-4o"]

# Define training configurations
training_configs = {
    "boostrapfewshot": {
        "optimizer": BootstrapFewShot(metric=metrics.get_score, max_rounds=4),
        "lm": lm_list,
    },
    "bootsrapfewshotwithrandomsearch": {
        "optimizer": BootstrapFewShotWithRandomSearch(
            metric=metrics.get_score,
            num_threads=1,
            num_candidate_programs=4,
            max_rounds=4,
        ),
        "lm": lm_list,
    },
    "miprov2fewshot": {
        "optimizer": MIPROv2(metric=metrics.get_score, num_threads=1, num_candidates=4),
        "lm": lm_list,
    },
}


def run_all_training_configurations():
    # Ensure evaluation directory exists
    eval_dir = Path("evaluations")
    eval_dir.mkdir(exist_ok=True)
    eval_file = eval_dir / "model_scores.csv"

    # Load existing scores if file exists
    if eval_file.exists():
        scores_df = pd.read_csv(eval_file)
    else:
        scores_df = pd.DataFrame(columns=["model_name", "score"])

    for run_name, config in training_configs.items():
        for lm_name in config["lm"]:
            model_name = f"penprompter-{run_name}-{lm_name}"
            print(f"\n{'='*50}")
            print(f"Starting training run: {model_name}")
            print(f"{'='*50}")

            try:
                # Create new HyperParams for this run
                current_params = HyperParams(
                    training_run_name=model_name,
                    fc_weights_path="models/factual_consistency/2024-11-01_bootsrapfewshotwithrandomsearch_second.json",
                )

                # Initialize language model
                _ = LanguageModel(lm=dspy.LM(f"openai/{lm_name}"))

                # Create new Trainer instance
                trainer = Trainer(
                    params=current_params,
                    network=network,
                    data=data,
                    optimizer=config["optimizer"],
                )

                # Train the model
                trainer.optimize()

                # Evaluate the model
                print("\nEvaluating model...")
                _, valid_path = get_train_and_valid_path()
                with open(valid_path, "rb") as f:
                    valid_dataset = pickle.load(f)
                scores = []
                for example in valid_dataset:
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
                        if score is not None:
                            scores.append(score)
                    except Exception as e:
                        print(f"Error in evaluation: {str(e)}")
                        continue

                mean_score = np.mean(scores) if scores else 0.0
                print(f"\nValidation Score: {mean_score:.4f}")

                # Save score to DataFrame
                new_score = pd.DataFrame(
                    {"model_name": [model_name], "score": [mean_score]}
                )
                scores_df = pd.concat([scores_df, new_score], ignore_index=True)

                # Save updated scores
                scores_df.to_csv(eval_file, index=False)

                print(f"\nSuccessfully completed training run: {model_name}")

            except Exception as e:
                print(f"\nError in training run {model_name}: {str(e)}")
                continue


if __name__ == "__main__":
    run_all_training_configurations()
