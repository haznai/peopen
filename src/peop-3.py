# %% imports
# types and attributes
# misc
import os
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal, Mapping, Optional, Union

# dspy imports
import dspy
from dspy.datasets import DataLoader
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, MIPROv2

# metrics
from evaluate import load
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk

# phoenix setup
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from rouge import Rouge


# %% Class definitions
class TrainingLanguage(Enum):
    GERMAN = "de"
    FRENCH = "fr"
    ITALIAN = "it"
    ALL = "all"


class ModelType(Enum):
    TEXT = "text"
    CHAT = "chat"


@dataclass(frozen=True)
class HyperParams:
    """
    Hyperparameters for the training, load once at the start of the program, never changed
    """

    # if data should be truncated
    is_truncation_required: bool = field(default=False)
    # how much of the training data should be used
    training_set_limit: Optional[int] = field(default=None)
    # how many validation examples should be used
    valid_set_limit: Optional[int] = field(default=None)
    # dataset training language
    language: TrainingLanguage = field(default=TrainingLanguage.ALL)
    # naming of run
    training_run_name: str = field(default="no_training_run_name_specified")

    # @todo: implement
    # not implemented yet:
    # phoenix_metadata: Optional[dict] = field(default=None)
    # model: str = field(default=None)
    # temperature: Optional[float] = field(default=None)
    # optimizer_params: Optional[dict] = field(default=None)


@dataclass(frozen=True)
class LanguageModel:
    """
    Defines which LM to use and configures dspy to use the model
    """

    name: str = field(default="")
    url: str = field(default="http://localhost:8080/v1/")
    api_key: str = field(default="no_api_key_specified")
    type: ModelType = field(default=ModelType.CHAT)
    model: dspy.OpenAI = field(init=False)

    def __post_init__(self):
        openai_model = dspy.OpenAI(
            model=self.name,
            api_key=self.api_key,
            api_base=self.url,
            model_type=self.type.value,
            max_tokens=8180,
        )
        dspy.settings.configure(lm=openai_model)
        object.__setattr__(self, "model", openai_model)


@dataclass(frozen=True)
class Logger:
    """
    Logging setup
    """

    endpoint: str = field(default="http://localhost:6006/v1/traces")

    def __post_init__(self):
        # phoenix setup to works seamlessly with dspy
        resource = Resource(attributes={})
        tracer_provider = trace_sdk.TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter(self.endpoint))
        )
        trace_api.set_tracer_provider(tracer_provider=tracer_provider)
        DSPyInstrumentor().instrument()

        # turn off local `INFO` level logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("root").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)


@dataclass(frozen=True)
class Dataset:
    """
    Dataset the model is trained on, preprocessed based on the hyperparameters
    """

    data: Mapping[Literal["train", "validation", "test"], list[dspy.Example]]

    def __init__(self, parameters: HyperParams):
        fields_to_use = ("text", "regeste", "language")  # type: ignore -- dspy bug
        input = ("text",)
        dataset = DataLoader().from_huggingface(
            dataset_name="rcds/swiss_leading_decision_summarization",
            fields=fields_to_use,  # type:ignore -- dspy bug
            input_keys=input,
            trust_remote_code=True,
        )

        # filter out irrelevant languages
        match parameters.language:
            case TrainingLanguage.ALL:
                pass
            # default case
            case _:
                for data_split, examples in dataset.copy().items():  # type: ignore
                    dataset[data_split] = list(  # type: ignore
                        filter(
                            lambda example: example["language"]
                            == parameters.language.value,
                            examples,
                        )
                    )

        # @todo: preprocess all datasets and ensure that labels and inputs are always
        # correct and you don't need to manually do it in the code
        # remove language key
        # this input handling is a mess atm
        for data_split, examples in dataset.copy().items():  # type: ignore
            dataset[data_split] = [  # type: ignore
                example.without("language").with_inputs("text") for example in examples
            ]

        # @todo: add truncation
        if parameters.is_truncation_required:
            pass

        object.__setattr__(self, "data", dataset)


@dataclass(frozen=True)
class Metrics:
    """
    Defines calculation of eval metrics
    """

    @staticmethod
    def _calculate_mean_bert_score(
        pred: str,
        ground_truth: str,
    ) -> float:
        """Computes and returns the average f1 BERTScore based on untokenized inputs"""
        bertscore = load("bertscore")
        bert_scores = bertscore.compute(
            predictions=[pred],
            references=[ground_truth],
            model_type="bert-base-multilingual-cased",
            lang=["de"],
        )
        # bertscore can return multiple values, so we average them
        return sum(bert_scores["f1"]) / len(bert_scores["f1"])  # type: ignore

    @staticmethod
    def _calculate_mean_rouge_score(
        pred: str,
        ground_truth: str,
    ) -> float:
        """
        Computes and returns the average ROUGE score based on untokenized inputs
        """
        rouge = Rouge()
        scores = rouge.get_scores(pred, ground_truth)[0]

        # average f1 of all ROUGE scores
        avg_score_through_all_metrics = 0
        num_scores = len(scores)
        for metric_name, metric_scores in scores.items():
            # average of f1-scores
            avg_score_through_all_metrics += metric_scores["f"]  # type: ignore

        avg_score_through_all_metrics = avg_score_through_all_metrics / num_scores
        return avg_score_through_all_metrics

    @staticmethod
    def get_score(example, prediction, trace=None):
        # function signature is important to be compatible with dspy optimizers
        # @todo: implement more metrics
        # @todo: make `regeste` not hard coded?

        return Metrics._calculate_mean_bert_score(
            pred=prediction.regeste, ground_truth=example.regeste
        )

        # return Metrics._calculate_mean_rouge_score(
        #     pred=prediction.regeste, ground_truth=example.regeste
        # )


class Network(dspy.Module):
    """
    Defines the model structure and training steps
    """

    def __init__(self):
        super().__init__()
        #### definition of modules ####
        self.text_splitter = dspy.TypedPredictor(
            self.TextSplitterSignature,
        )
        self.sachverhalt_summarizer = dspy.TypedPredictor(
            self.SachverhaltSummarizerSignature
        )
        self.erwaehgungen_summarizer = dspy.TypedPredictor(
            self.ErwaehgungenSummarizerSignature
        )

        self.regeste_generator = dspy.TypedChainOfThought(
            self.RegesteGeneratorSignature,
        )

    def forward(self, text: str):
        # @todo: implement phoenix attributes/metadata here @see first notebook

        #### execution and passing of data ####
        # text parameter is equivalent to the lds dataset `text` input
        splitted_results = self.text_splitter(text=text)
        gekuerzter_sachverhalt = self.sachverhalt_summarizer(
            sachverhalt=splitted_results.sachverhalt
        ).gekuerzter_sachverhalt

        gekuerzte_erwaehgungen = self.erwaehgungen_summarizer(
            erwaehgungen=splitted_results.erwaehgungen
        ).gekuerzte_erwaehgungen

        return self.regeste_generator(
            gekuerzter_sachverhalt=gekuerzter_sachverhalt,
            gekuerzte_erwaehgungen=gekuerzte_erwaehgungen,
            dispositiv=splitted_results.dispositiv,
        )

    class TextSplitterSignature(dspy.Signature):
        """
        Ein Schweizer Gerichtsurteil setzt sich aus Sachverhalt, Erwägungen und Dispositiv zusammen. Teile den Text des Gerichturteils auf in Sachverhalt, Erwägungen und Dispositiv ohne Zensur/Modifikation.
        """

        text = dspy.InputField()
        sachverhalt = dspy.OutputField()
        erwaehgungen = dspy.OutputField()
        dispositiv = dspy.OutputField()

    class SachverhaltSummarizerSignature(dspy.Signature):
        """
        Lies und verstehe den gegebenen Sachverhalt und erstelle dann eine Kurzfassung
        """

        sachverhalt = dspy.InputField()
        gekuerzter_sachverhalt = dspy.OutputField()

    class ErwaehgungenSummarizerSignature(dspy.Signature):
        """
        Analysiere die Erwägungen, um die Hauptargumente und Gründe zu identifizieren und erstelle dann eine Kurzfassung
        """

        erwaehgungen = dspy.InputField()
        gekuerzte_erwaehgungen = dspy.OutputField()

    class RegesteGeneratorSignature(dspy.Signature):
        """
        Das Gerichtsurteil ist ein grosser Fliesstext. Ein Schweizer Gerichtsurteil setzt sich aus Sachverhalt, Erwägungen und Dispositiv zusammen. Ich gebe dir den grossen Fliesstext, und gekürzte Versionen des Sachverhalts und der Erwägungen.

        Erstelle mir basierend auf den Fliesstext und den gekürzten Versionen eine Regeste. Die Regeste sollte aus drei sehr kurzen Teilen bestehen: 1. Zitiere die wichtigsten relevanten Artikelziffern (ohne den Artikeltitel). 2. Nenne kurze, relevante, deskriptive Keywords, über die Thematik des Falls. 3. sehr kurzen Fliesstext mit einem Urteil.

        Die Regeste sollte eine klare und strukturierte Kurzzusammenfassung des Urteils bieten, die aus zitierten Artikeln, Keywords und einem sehr kurzer Fliesstext besteht. Beachte das Dispositiv, da es das endgültige Urteil enthält.
        """

        gekuerzter_sachverhalt = dspy.InputField()
        gekuerzte_erwaehgungen = dspy.InputField()
        dispositiv = dspy.InputField()

        regeste = dspy.OutputField()


@dataclass(frozen=True)
class Trainer:
    """
    Optimizes network, actual training happens here
    """

    params: HyperParams
    network: Network
    data: Dataset
    optimizer: Union[BootstrapFewShotWithRandomSearch, MIPROv2]

    # @todo implement for optimizing based on hyperparams as well

    def optimize(self, **kwargs):
        # @todo: this operation happens wayyyy too late, can be done earlier
        trainset = self.data.data["train"]
        valset = self.data.data["validation"]

        if self.params.training_set_limit is not None:
            trainset = trainset[: self.params.training_set_limit]

        if self.params.valid_set_limit is not None:
            valset = valset[: self.params.valid_set_limit]

        optimized_network = self.optimizer.compile(
            trainset=trainset,
            valset=valset,
            **kwargs,
        )

        # saving of the optimized model
        iso_date = datetime.today().strftime("%Y-%m-%d")
        file_path = f"models/{iso_date}_{self.params.training_run_name}"
        file_extension = ".json"

        # check if file exists, if yes, append a character until filename doesn't exist
        while os.path.exists(file_path + file_extension):
            file_path += "_new"

        optimized_network.save(file_path + file_extension)


# %% Loading in the data

# @todo: define everything outside of this notebook, with all parameters load into from outside this file
# @todo: just pass params to the trainer class, all other classes should not need to know about it
params = HyperParams(
    training_run_name="second_training_run",
    language=TrainingLanguage.GERMAN,
    training_set_limit=100,
    valid_set_limit=50,
)


lm = LanguageModel(
    name="gpt-4o-mini-2024-07-18",
    url="https://api.openai.com/v1/",
    api_key=os.environ.get("OPENAI_API_KEY"),  # type: ignore
)
_ = Logger()  # just needs to be init
data = Dataset(parameters=params)
metrics = Metrics()
network = Network()
# this optimizer is known to work, but in preliminary testing, turned out to be dissapointing


# @todo: fix regex bug https://github.com/stanfordnlp/dspy/blob/main/dspy/propose/grounded_proposer.py
# @todo: fix not saving bug (might be related to the above)
# @todo: look into extra options for fields: Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Gekuerzter Sachverhalt:', 'desc': '${gekuerzter_sachverhalt}', 'format': <function TypedPredictor._prepare_signature.<locals>.<lambda> at 0x31ab579c0>})\n    gekuerzte_erwaehgungen = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Gekuerzte Erwaehgungen:', 'desc': '${gekuerzte_erwaehgungen}', 'format': <function TypedPredictor._prepare_signature.<locals>.<lambda> at 0x31b003d80>})\n    dispositiv = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Dispositiv:', 'desc': '${dispositiv}', 'format': <function TypedPredictor._prepare_signature.<locals>.<lambda> at 0x31b003060>})\n    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': \"Reasoning: Let's think step by step in order to\", 'desc': '${produce the regeste}. We ...', '__dspy_field_type': 'output', 'format': <function TypedPredictor._prepare_signature.<locals>.<lambda> at 0x31b002b60>, 'parser': <class 'str'>})\n    regeste = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Regeste:', 'desc': '${regeste}', 'format': <function TypedPredictor._prepare_signature.<locals>.<lambda> at 0x31b001440>, 'parser': <class 'str'>})\n)"

# optimizer = MIPROv2(
#     prompt_model=lm.model,
#     task_model=lm.model,
#     metric=metrics.get_score,
#     init_temperature=0.5,
#     log_dir="models",
# )

optimizer = BootstrapFewShotWithRandomSearch(
    metric=metrics.get_score,
    max_rounds=3,
)

trainer = Trainer(params=params, network=network, data=data, optimizer=optimizer)

# %% training
# @todo rethink the abstraction with `extra_params` passing to `Trainer` (and passing `params` earlier)
extra_params = {
    "student": network,
    # "num_batches": 5,
}


trainer.optimize(**extra_params)
