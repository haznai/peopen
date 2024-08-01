import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md("""# Import""")
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
    from dspy.predict import Predict
    from dspy.datasets import DataLoader
    from dspy.teleprompt.teleprompt import Teleprompter
    return (
        DataLoader,
        Enum,
        List,
        Optional,
        Predict,
        Teleprompter,
        dataclass,
        dspy,
        field,
        mo,
    )


@app.cell
def __(mo):
    mo.md(r"""# imperative shell""")
    return


@app.cell
def __(mo):
    mo.md(r"""## Class definitions""")
    return


@app.cell
def __(
    DataLoader,
    Enum,
    Literal,
    Mapping,
    Optional,
    Predictor,
    Teleprompter,
    dataclass,
    dspy,
    field,
    load,
):
    class TrainingLanguage(Enum):
        GERMAN = "de"
        FRENCH = "fr"
        ITALIAN = "it"
        ALL = "all"


    class ModelType(Enum):
        TEXT = "text"
        CHAT = "chat"


    @dataclass(frozen=True)
    class LanguageModel:
        """
        Defines which LM to use and configures dspy to use the model
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

        # if data should be truncated
        is_truncation_required: bool = field(default=False)
        # how much of the training data should be used
        training_set_limit: Optional[int] = field(default=None)
        # dataset training language
        language: TrainingLanguage = field(default=TrainingLanguage.ALL)

        # @todo: implement
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


    @dataclass(frozen=True)
    class Metrics:
        """
        Defines calculation of eval metrics
        """

        @staticmethod
        def _calculate_bert_score(
            pred: str,
            ground_truth: str,
        ) -> float:
            """Computes and returns the average f1 BERTScore based on untokenized inputs"""
            bertscore = load("bertscore")
            bert_scores = bertscore.compute(
                predictions=[pred],
                references=[ground_truth],
                model_type="bert-base-multilingual-cased",
                lang=["de", "fr", "it"],
            )
            # bertscore can return multiple values, so we average them
            return sum(bert_scores["f1"]) / len(bert_scores["f1"])

        @staticmethod
        def get_metrics(example, prediction, trace=None):
            # function signature is important to be compatible with dspy optimizers
            # @todo: implement more metrics
            # @todo: make `regeste` not hard coded?
            return Metrics._calculate_bert_score(
                pred=prediction.regeste, ground_truth=example.regeste
            )


    class Network(dspy.Module):
        """
        Defines the model structure and training steps
        """

        prediction_pipeline: Predictor = None

        # @todo: split up
        class DecisionSummarizationSignature(dspy.Signature):
            """Ziel: Generiere eine Regeste basierend auf einem Schweizer Gerichtsurteils.\nHintergrund: Ein Schweizer Gerichtsurteil setzt sich aus Sachverhalt, Erwägungen und Dispositiv zusammen. Die Regeste dient als Kurzzusammenfassung und beinhaltet Leitsätze des Urteils. Nur Leitentscheide haben eine Regeste.\nAnweisung:\n1. Sachverhalt: Lies und verstehe den gegebenen Sachverhalt.\n2. Erwägungen: Analysiere die Erwägungen, um die Hauptargumente und Gründe zu identifizieren.\n3. Dispositiv: Beachte das Dispositiv, da es das endgültige Urteil enthält.\n4. Erstelle die Regeste: Die Regeste sollte aus drei sehr kurzen Teilen bestehen: a. Zitiere die wichtigsten relevanten Artikelziffern (ohne den Artikeltitel). b. Nenne kurze, relevante, deskriptive Keywords, über die Thematik des Falls. c. Formuliere einen sehr kurzen Fliesstext, der die wichtigsten Erwägungen zitiert und kurz zusammenfasst.\nOutput: Die Regeste sollte eine klare und strukturierte Kurzzusammenfassung des Urteils bieten, die aus zitierten Artikeln, Keywords und einem sehr kurzen Fliesstext besteht."""

            text = dspy.InputField()
            regeste = dspy.OutputField()

        def __init__(self):
            super().__init__()
            # @todo: change up prediction pipeline after splitting
            self.prediction_pipeline = dspy.ChainOfThought(
                self.DecisionSummarizationSignature()
            )

        def forward(self, text: str):
            assert prediction_pipeline
            # @todo: implement phoenix attributes/metadata here @see first notebook
            # @todo: make this universal so `text` isn't hardcoded?
            # text parameter is equivalent to the lds dataset `text` input
            return prediction_pipeline(text=text)


    @dataclass(frozen=True)
    class Trainer:
        """ """

        metrics: Metrics
        network: Network
        data: Dataset
        optimizer: Teleprompter
        # @todo implement for optimizing based on hyperparams as well
        # hyperparameters: HyperParams

        def optimize(self):
            _save_model()

        def _save_model(self):
            pass
    return (
        Dataset,
        HyperParams,
        LanguageModel,
        Metrics,
        ModelType,
        Network,
        Trainer,
        TrainingLanguage,
    )


@app.cell
def __(mo):
    mo.md(
        r"""

        ## Instantiation of all objects
        """
    )
    return


@app.cell
def __():
    # @todo: define everything outside of this notebook, with all parameters load into from outside this file


    # LanguageModel:
    # HyperParams:
    # Dataset:
    # Metrics:
    # Network
    # Trainer:
    return


@app.cell
def __(mo):
    mo.md(r"""# Functional core""")
    return


@app.cell
def __(mo):
    mo.md(r"""## Testing the model with a call""")
    return


if __name__ == "__main__":
    app.run()
