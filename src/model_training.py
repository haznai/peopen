# imports
from model_definition import (
    HyperParams,
    Dataset,
    LanguageModel,
    Logger,
    Metrics,
    Network,
    Trainer,
)

import os
from typing import Tuple
from pathlib import Path
from dspy import BootstrapFewShotWithRandomSearch


# %% Helper functions
def get_train_and_valid_path() -> Tuple[Path, Path]:
    """
    Get the project root directory.

    This function works in both script and Jupyter notebook environments.
    It tries different methods to find the project root.

    Returns:
        Path: The path to the project root directory.
    """
    try:
        # Try to get the path of the current file (works in scripts)
        path = Path(__file__).resolve().parent.parent
    except NameError:
        try:
            # Try to get the path from Jupyter notebook
            import IPython

            path = Path(IPython.get_ipython().kernel.profile_dir).parent.parent  # type: ignore
        except Exception:
            # Fallback to current working directory
            path = Path(os.getcwd()).parent.resolve()

    train_path = path.joinpath(
        "data", "processed", "volksiniativen_with_wortlaut_dspy_dataset_train.pkl"
    )

    valid_path = path.joinpath(
        "data", "processed", "volksiniativen_with_wortlaut_dspy_dataset_valid.pkl"
    )

    return (train_path, valid_path)


# %% Loading in the data
hyper_params = HyperParams(training_run_name="first_run_2024-10-06")
lm = LanguageModel(
    name="gpt-4o-mini-2024-07-18",
    url="https://api.openai.com/v1/",
    api_key=os.environ.get("OPENAI_API_KEY"),  # type: ignore
)
_ = Logger()  # just needs to be init


train_path, valid_path = get_train_and_valid_path()

data = Dataset(train_pickle_path=str(train_path), valid_pickle_path=str(valid_path))


metrics = Metrics()
network = Network()
trainer = Trainer(
    params=hyper_params,
    network=network,
    data=data,
    optimizer=BootstrapFewShotWithRandomSearch(metric=metrics.get_score, num_threads=1),
)


# %% Training the model
trainer.optimize()
