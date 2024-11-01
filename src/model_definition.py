## %% imports
# types and attributes
# misc
import logging
import os
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Literal, Mapping, Optional, Tuple

# dspy imports
import dspy
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPROv2
from sympy.printing.numpy import S

# custom models
from src.submodels.factual_consistency_model import FactualConsistencyNetwork
from src.submodels.article_number_retrieval_model import ArticleNumberRM

# metrics
from evaluate import EvaluationModule, load
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk

# phoenix setup
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from rouge import Rouge


# %% Class definitions
@dataclass(frozen=True)
class HyperParams:
    """
    Hyperparameters for the training, loaded once at the start of the program.
    """

    # How much of the training data should be used
    training_set_limit: Optional[int] = field(default=None)
    # How many validation examples should be used
    valid_set_limit: Optional[int] = field(default=None)
    # Naming of the training run
    training_run_name: str = field(default="no_training_run_name_specified")
    # Other hyperparameters
    phoenix_metadata: Optional[dict] = field(default=None)
    # path to factual consistency model weights
    fc_weights_path: str = field(default="no_factual_consistency_model_path_specified")


class ModelType(Enum):
    TEXT = "text"
    CHAT = "chat"


@dataclass(frozen=True)
class LanguageModel:
    """
    Defines which language model to use and configures dspy to use the model.
    """

    name: str = field(default="")
    url: str = field(default="http://localhost:8080/v1/")
    api_key: str = field(default="no_api_key_specified")
    type: ModelType = field(default=ModelType.CHAT)
    max_tokens: int = field(default=8180)
    model: dspy.OpenAI = field(init=False)

    def __post_init__(self):
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm)
        object.__setattr__(self, "model", lm)


@dataclass(frozen=True)
class Logger:
    """
    Logging setup.
    """

    training_run_name: str = field(default="no_training_run_name_specified")
    endpoint: str = field(default="http://localhost:6006/v1/traces")

    def __post_init__(self):
        # Phoenix setup to work seamlessly with dspy
        resource = Resource(attributes={"training_run_name": self.training_run_name})
        tracer_provider = trace_sdk.TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter(self.endpoint))
        )
        trace_api.set_tracer_provider(tracer_provider=tracer_provider)
        DSPyInstrumentor().instrument()

        # Turn off local `INFO` level logging to not clutter the logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("root").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)


@dataclass(frozen=True)
class Dataset:
    """
    Dataset the model is trained on, with split of training and validation data.
    """

    data: Mapping[Literal["train", "validation"], List[dspy.Example]]

    def __init__(
        self,
        train_pickle_path: str,
        valid_pickle_path: str,
    ):
        with open(train_pickle_path, "rb") as f:
            train_dataset = pickle.load(f)

            if (train_dataset is None) or (len(train_dataset) == 0):
                raise ValueError("Training dataset is empty or wrongly loaded")

        with open(valid_pickle_path, "rb") as f:
            valid_dataset = pickle.load(f)

            if (valid_dataset is None) or (len(valid_dataset) == 0):
                raise ValueError("Validation dataset is empty or wrongly loaded")

        dataset = {
            "train": train_dataset,
            "validation": valid_dataset,
        }

        object.__setattr__(self, "data", dataset)


@dataclass(frozen=True)
class Metrics:
    """
    Defines calculation of evaluation metrics.
    """

    evaltuator_bertscore = load("bertscore", keep_in_memory=True)

    @staticmethod
    def _calculate_mean_bert_score(
        evaluator: EvaluationModule,
        pred: str,
        ground_truth: str,
    ) -> float:
        """Computes and returns the average f1 BERTScore based on untokenized inputs"""

        bert_scores = evaluator.compute(
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

    def get_score(self, example, prediction, trace=None):
        # function signature is important to be compatible with dspy optimizers
        # @todo: implement more metrics
        # @todo: make `Wortlaut` not hard coded?

        return Metrics._calculate_mean_bert_score(
            evaluator=self.evaltuator_bertscore,
            pred=prediction.final_wortlaut,
            ground_truth=example.Wortlaut,
        )

        # @todo: ayo what's goin' on here
        # return Metrics._calculate_mean_rouge_score(
        #     pred=prediction.regeste, ground_truth=example.regeste
        # )


class PenPrompterNetwork(dspy.Module):
    """
    Defines the model structure and training steps.
    """

    fc_model: FactualConsistencyNetwork

    class WriteHumanUnderstandableDraftSignature(dspy.Signature):
        """
        Create a simple, human-understandable draft of the popular initiative.

        This module takes the input information and produces a clear, straightforward
        version of the initiative. The output should be easily understood by the general
        public, avoiding legal jargon and complex language. Focus on presenting the main
        ideas and objectives in simple terms.
        """

        summarized_im_detail = dspy.InputField()
        arguments_committee = dspy.InputField()
        arguments_bundesrat = dspy.InputField()

        human_understandable_draft = dspy.OutputField()

    class HighlightLegalRequirementsSignature(dspy.Signature):
        """
        Identify legal requirements, constitutional provisions, and mandatory elements for the final law text ('Wortlaut').
        """

        human_understandable_draft = dspy.InputField()

        legal_requirements = dspy.OutputField()

    class GenerateDraftWortlautSignature(dspy.Signature):
        """
        Generate a draft Wortlaut (legal text) of the proposed constitutional amendment.

        Importantly, instead of Article numbers in the legal text, placeholders should be used. Articles with placeholders should always look like this: 'Art. [ARTICLE NUMBER HERE]'
        """

        human_understandable_draft = dspy.InputField()
        legal_requirements = dspy.InputField()

        draft_wortlaut_without_article_numbers = dspy.OutputField()

    class GenerateFinalWortlautSignature(dspy.Signature):
        """
        Insert the article numbers into the 'Art. [ARTICLE NUMBER HERE]' placeholders in the draft and output the final legal text.
        """

        draft_wortlaut_without_article_numbers = dspy.InputField()
        article_numbers = dspy.InputField()

        final_wortlaut = dspy.OutputField()

    def __init__(self, fc_model: FactualConsistencyNetwork):
        assert fc_model._compiled, "FactualConsistencyNetwork must be compiled before initializing PenPrompterNetwork"

        super().__init__()

        self.fc_model = fc_model

        self.highlight_legal_requirements_module = dspy.TypedChainOfThought(
            self.HighlightLegalRequirementsSignature,
        )

        self.write_human_understandable_draft_module = dspy.TypedChainOfThought(
            self.WriteHumanUnderstandableDraftSignature,
        )

        self.generate_draft_wortlaut_module = dspy.TypedChainOfThought(
            self.GenerateDraftWortlautSignature,
        )

        self.generate_final_wortlaut_module = dspy.TypedPredictor(
            self.GenerateFinalWortlautSignature,
        )

        self.article_number_retriever = ArticleNumberRM()

    def get_first_draft(
        self,
        titel,
        im_detail,
        argumenteKomitee,
        empfehlungKomitee,
        argumenteBundesrat,
        empfehlungBundesrat,
    ):
        summarizations = self.fc_model(
            titel,
            im_detail,
            argumenteKomitee,
            empfehlungKomitee,
            argumenteBundesrat,
            empfehlungBundesrat,
        )
        human_understandable_draft = self.write_human_understandable_draft_module(
            summarized_im_detail=summarizations.summarized_im_detail,
            arguments_committee=summarizations.arguments_committee,
            arguments_bundesrat=summarizations.arguments_bundesrat,
        ).human_understandable_draft

        return human_understandable_draft

    def get_final_prediction(self, human_understandable_draft: str):
        legal_requirements = self.highlight_legal_requirements_module(
            human_understandable_draft=human_understandable_draft,
        ).legal_requirements

        draft_wortlaut_without_article_numbers = self.generate_draft_wortlaut_module(
            human_understandable_draft=human_understandable_draft,
            legal_requirements=legal_requirements,
        ).draft_wortlaut_without_article_numbers

        article_numbers = self.article_number_retriever(
            draft_wortlaut_without_article_numbers
        ).retrieved_article_numbers  # type: ignore

        article_numbers = f"article_numbers={article_numbers}"

        return self.generate_final_wortlaut_module(
            draft_wortlaut_without_article_numbers=draft_wortlaut_without_article_numbers,
            article_numbers=article_numbers,
        )

    def forward(
        self,
        titel,
        im_detail,
        argumenteKomitee,
        empfehlungKomitee,
        argumenteBundesrat,
        empfehlungBundesrat,
    ):
        return self.get_final_prediction(
            self.get_first_draft(
                titel,
                im_detail,
                argumenteKomitee,
                empfehlungKomitee,
                argumenteBundesrat,
                empfehlungBundesrat,
            )
        )


@dataclass(frozen=True)
class Trainer:
    """
    Optimizes the network; actual training happens here.
    """

    params: HyperParams
    network: dspy.Module
    data: Dataset
    optimizer: BootstrapFewShot | BootstrapFewShotWithRandomSearch | MIPROv2

    def optimize(
        self,
    ):
        trainset = self.data.data["train"]
        valset = self.data.data["validation"]

        if self.params.training_set_limit is not None:
            trainset = trainset[: self.params.training_set_limit]

        if self.params.valid_set_limit is not None:
            valset = valset[: self.params.valid_set_limit]

        if isinstance(self.optimizer, BootstrapFewShot):
            optimized_network = self.optimizer.compile(
                student=self.network,
                trainset=trainset,
            )
        elif isinstance(self.optimizer, BootstrapFewShotWithRandomSearch):
            optimized_network = self.optimizer.compile(
                student=self.network, trainset=trainset, valset=valset
            )
        elif isinstance(self.optimizer, MIPROv2):
            optimized_network = self.optimizer.compile(
                student=self.network,
                trainset=trainset,
                valset=valset,
                minibatch_size=2,
                requires_permission_to_run=False,
            )

        # Saving the optimized model
        iso_date = datetime.today().strftime("%Y-%m-%d")
        file_path = f"models/{iso_date}_{self.params.training_run_name}"
        file_extension = ".json"

        # Check if file exists; if yes, append a character until filename doesn't exist
        while os.path.exists(file_path + file_extension):
            file_path += "_new"

        optimized_network.save(file_path + file_extension)


# %% Helper functions
def get_train_and_valid_path() -> Tuple[Path, Path]:
    """
    Get the project root directory.

    This function works in both script and Jupyter notebook environments.
    It tries different methods to find the project root.

    Returns:
        Tuple with paths to the train and validation datasets
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

    train_path = path.joinpath("data", "processed", "final_truncated_train_dataset.pkl")

    valid_path = path.joinpath("data", "processed", "final_truncated_valid_dataset.pkl")

    return (train_path, valid_path)


def get_model_path(model_json_path: str) -> Path:
    """
    Only pass the name of the saved model (+ .json file extension).

    This function works in both script and Jupyter notebook environments.

    Returns:
        Path: The path to the model file.
    """
    path = None
    try:
        # Try to get the path of the root
        path = Path(__file__).resolve().parent.parent
    except Exception:
        try:
            import IPython

            path = Path(IPython.get_ipython().kernel.profile_dir).parent.parent  # type: ignore
        except Exception:
            path = Path(os.getcwd()).parent.resolve()

    if path is None:
        raise Exception(f"model with path {model_json_path} couldn't be initialized")

    model_path = path.joinpath("models", model_json_path)

    if not model_path.exists():
        raise Exception(f"model with path {model_path} couldn't be initialized")

    return model_path
