import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md("""# Imports""")
    return


@app.cell
def __():
    # marimo notebook
    import marimo as mo

    # types and attributes
    from enum import Enum
    from dataclasses import dataclass, field
    from typing import List, Optional

    # dspy imports
    import dspy
    from dspy.datasets import DataLoader
    return DataLoader, Enum, List, Optional, dataclass, dspy, field, mo


@app.cell
def __(mo):
    mo.md(r"""# imperative shell""")
    return


@app.cell
def __(mo):
    mo.md(r"""## Class definitions""")
    return


@app.cell
def __(Enum):
    class TrainingLanguage(Enum):
        GERMAN = "de"
        FRENCH = "fr"
        ITALIAN = "it"
        ALL = "all"


    class ModelType(Enum):
        TEXT = "text"
        CHAT = "chat"
    return ModelType, TrainingLanguage


@app.cell
def __(
    DataLoader,
    Literal,
    Mapping,
    ModelType,
    Optional,
    TrainingLanguage,
    dataclass,
    dspy,
    field,
):
    @dataclass(frozen=True)
    class Model:
        """
        Defines which model to use
        """

        name: str = field(default="")
        url: str = field(default="http://localhost:8080/v1/")
        api_key: str = field(default="")
        type: ModelType = field(default=ModelType.CHAT)
        model: dspy.OpenAI = field(default=None)

        def __post_init__(self):
            openai_model = dspy.OpenAI(
                model=self.name, api_base=self.url, model_type=self.type.value
            )
            dspy.settings.configure(lm=openai_model)
            object.__setattr__(self, "model", openai_model)


    @dataclass(frozen=True)
    class HyperParams:
        """
        Hyperparameters for the training, load once at the start of the program, never changed
        """

        # if data should be truncate
        is_truncation_required: bool = field(default=False)
        # how much of the training data should be used
        training_set_limit: Optional[int] = field(default=None)
        # dataset training language
        language: TrainingLanguage = field(default=TrainingLanguage.ALL)

        # not implemented yet:
        # phoenix_metadata: Optional[dict] = field(default=None)
        # model: str = field(default=None)
        # temperature: Optional[float] = field(default=None)
        # optimizer_params: Optional[dict] = field(default=None)


    @dataclass(frozen=True)
    class Dataset:
        """
        Dataset the model is trained on, preprocessed based on the hyperparameters
        """

        data: Mapping[Literal["train", "validation", "test"], list[dspy.Example]]

        def __init__(self, parameters: HyperParams):
            # load data from huggingface
            fields_to_use = ("text", "regeste", "language")
            input = ("text",)
            DataLoader.from_huggingface
            dataset = DataLoader().from_huggingface(
                dataset_name="rcds/swiss_leading_decision_summarization",
                fields=fields_to_use,
                input_keys=input,
            )

            # filter out irrelevant languages
            match parameters.language:
                case TrainingLanguage.ALL:
                    pass
                # default case
                case _:
                    for data_split, examples in dataset.copy().items():
                        dataset[data_split] = list(
                            filter(
                                lambda example: example["language"]
                                == parameters.language.value,
                                examples,
                            )
                        )

            # @todo: add truncation
            if parameters.is_truncation_required:
                pass

            # @todo: implement training data limit
            if parameters.training_set_limit is not None:
                pass

            object.__setattr__(self, "data", dataset)
    return Dataset, HyperParams, Model


@app.cell
def __(mo):
    mo.md(r"""## Loading the data""")
    return


@app.cell
def __(Dataset, HyperParams, TrainingLanguage):
    # @todo: change so this is read from outside the code
    hyperparameters = HyperParams(
        is_truncation_required=False,
        training_set_limit=None,
        language=TrainingLanguage.GERMAN,
    )
    dataset = Dataset(hyperparameters).data
    return dataset, hyperparameters


@app.cell
def __(mo):
    mo.md(r"""## Loading the model""")
    return


@app.cell
def __(Model):
    model = Model().model
    return model,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
