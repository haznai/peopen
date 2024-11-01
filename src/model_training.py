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


import os
from dspy import BootstrapFewShotWithRandomSearch
import dspy


# %%# %% Loading in the data
train_path, valid_path = get_train_and_valid_path()

hyper_params = HyperParams(
    training_run_name="penprompter-first-full-training",
    fc_weights_path="models/factual_consistency/2024-11-01_bootsrapfewshotwithrandomsearch_second.json",
)
lm = LanguageModel(lm=dspy.LM("openai/gpt-4o-mini"))
_ = Logger()  # just needs to be init
data = Dataset(train_pickle_path=str(train_path), valid_pickle_path=str(valid_path))

metrics = Metrics()
fc_model = FactualConsistencyNetwork()
fc_model.load(hyper_params.fc_weights_path)
fc_model._compiled = True
network = PenPrompterNetwork(fc_model)
trainer = Trainer(
    params=hyper_params,
    network=network,
    data=data,
    optimizer=BootstrapFewShotWithRandomSearch(
        metric=metrics.get_score, num_threads=1, num_candidate_programs=4, max_rounds=4
    ),
)


# %% Training the model
trainer.optimize()
