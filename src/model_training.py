# imports
from model_definition import (
    HyperParams,
    Dataset,
    LanguageModel,
    Logger,
    Metrics,
    Network,
    Trainer,
    get_train_and_valid_path,
)

import os
from dspy import BootstrapFewShotWithRandomSearch


# %%# %% Loading in the data
train_path, valid_path = get_train_and_valid_path()

hyper_params = HyperParams(training_run_name="first_run_2024-10-06")
lm = LanguageModel(
    name="gpt-4o-mini-2024-07-18",
    url="https://api.openai.com/v1/",
    api_key=os.environ.get("OPENAI_API_KEY"),  # type: ignore
)
_ = Logger()  # just needs to be init
data = Dataset(train_pickle_path=str(train_path), valid_pickle_path=str(valid_path))
metrics = Metrics()
network = Network()
trainer = Trainer(
    params=hyper_params,
    network=network,
    data=data,
    optimizer=BootstrapFewShotWithRandomSearch(
        metric=metrics.get_score, num_threads=1, num_candidate_programs=4, max_rounds=2
    ),
)


# %% Training the model
trainer.optimize()
