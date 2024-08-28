# %% imports
# types and attributes
# misc
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal, Mapping, Optional

# dspy imports
import dspy
from dspy.datasets import DataLoader
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.teleprompt import Teleprompter

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
from transformers.models.dpt.modeling_dpt import Tuple


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


@dataclass(frozen=True)
class Dataset:
    """
    Dataset the model is trained on, preprocessed based on the hyperparameters
    """

    data: Mapping[Literal["train", "validation", "test"], list[dspy.Example]]

    def __init__(self, parameters: HyperParams):
        # load data from huggingface
        fields_to_use: Tuple[str] = ("text", "regeste", "language")
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
    def _calculate_rouge_score(
        pred: str,
        ground_truth: str,
    ) -> float:
        """
        Computes and returns the average ROUGE score based on untokenized inputs
        """
        rouge = Rouge()
        rouge.get_scores(pred, ground_truth)[0]

    @staticmethod
    def get_score(example, prediction, trace=None):
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

    def __init__(self):
        super().__init__()

    def forward(self, text: str):
        # @todo: implement phoenix attributes/metadata here @see first notebook

        #### definition of modules ####
        text_splitter = dspy.Predict(
            self.TextSplitterSignature,
        )
        sachverhalt_summarizer = dspy.Predict(self.SachverhaltSummarizerSignature)
        erwaehgungen_summarizer = dspy.Predict(self.ErwaehgungenSummarizerSignature)

        regeste_generator = dspy.Predict(
            self.RegesteGeneratorSignature,
        )

        #### execution and passing of data ####
        # text parameter is equivalent to the lds dataset `text` input
        splitted_results = text_splitter(text=text)
        gekuerzter_sachverhalt = sachverhalt_summarizer(
            sachverhalt=splitted_results.sachverhalt
        ).gekuerzter_sachverhalt

        gekuerzte_erwaehgungen = erwaehgungen_summarizer(
            erwaehgungen=splitted_results.erwaehgungen
        ).gekuerzte_erwaehgungen

        return regeste_generator(
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
    optimizer: Teleprompter
    # @todo implement for optimizing based on hyperparams as well
    # hyperparameters: HyperParams

    def optimize(self):
        # @todo: this operation happens wayyyy too late, can be done earlier
        trainset = self.data.data["train"]
        valset = self.data.data["validation"]
        if self.params.training_set_limit is not None:
            trainset = trainset[: self.params.training_set_limit]

        optimized_network = self.optimizer.compile(
            student=self.network,
            teacher=self.network,
            trainset=trainset,
            valset=valset,
        )

        # saving of the optimized model
        iso_date = datetime.today().strftime("%Y-%m-%d")
        file_path = f"models/{iso_date}_{self.params.training_run_name}"

        # check if file exists, if yes, append a character until filename doesn't exist
        while os.path.exists(file_path):
            file_path += "_new"

        file_extension = ".json"
        optimized_network.save(file_path + file_extension)


# %% Loading in the data

# @todo: define everything outside of this notebook, with all parameters load into from outside this file
params = HyperParams(
    training_run_name="first_training_run", language=TrainingLanguage.GERMAN
)
lm = LanguageModel(
    name="gpt-4o-mini-2024-07-18",
    url="https://api.openai.com/v1/",
    api_key="redacted",
)
logging = Logger()
data = Dataset(parameters=params)
metrics = Metrics()
network = Network()
# this optimizer is known to work, but in preliminary testing, turned out to be dissapointing
optimizer = BootstrapFewShotWithRandomSearch(
    metric=metrics.get_score, max_labeled_demos=8, num_candidate_programs=16
)


# this optimizer is the future of dspy-ai, but throws weird errors at times
# from dspy.teleprompt import MIPROv2
# optimizer = MIPROv2(prompt_model=lm.model, task_model=lm.model, metric=metrics.get_score)

trainer = Trainer(params=params, network=network, data=data, optimizer=optimizer)

# %% training
trainer.optimize()
