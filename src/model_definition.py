## %% imports
# types and attributes
# misc
import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Literal, Mapping, Optional

# dspy imports
import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

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


class ModelType(Enum):
    TEXT = "text"
    CHAT = "chat"


# @todo: switch to 2.5 dspy lm setup
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


class Network(dspy.Module):
    """
    Defines the model structure and training steps.
    """

    class SummarizeImDetailSignature(dspy.Signature):
        """
        Summarize the detailed explanations ('im_detail') to provide a concise overview of the initiative.
        """

        titel = dspy.InputField()
        im_detail = dspy.InputField()

        summarized_im_detail = dspy.OutputField()

    class ExtractArgumentsCommitteeSignature(dspy.Signature):
        """
        Extract the main arguments presented by the supporting committee and their recommendation.
        """

        titel = dspy.InputField()
        argumenteKomitee = dspy.InputField()
        empfehlungKomitee = dspy.InputField()

        arguments_committee = dspy.OutputField()

    class ExtractArgumentsFederalCouncilSignature(dspy.Signature):
        """
        Extract the main arguments presented by the Federal Council opposing the initiative and their recommendation.
        """

        titel = dspy.InputField()
        argumenteBundesrat = dspy.InputField()
        empfehlungBundesrat = dspy.InputField()

        arguments_bundesrat = dspy.OutputField()

    class ClarifyAmbiguousTermsSignature(dspy.Signature):
        """
        Identify and clarify ambiguous or unclear terms in the summaries and arguments.
        """

        summarized_im_detail = dspy.InputField()
        arguments_committee = dspy.InputField()
        arguments_bundesrat = dspy.InputField()

        clarified_terms = dspy.OutputField()

    class HighlightLegalRequirementsSignature(dspy.Signature):
        """
        Identify legal requirements, constitutional provisions, and mandatory elements for the 'Wortlaut'.
        """

        summarized_im_detail = dspy.InputField()
        clarified_terms = dspy.InputField()

        legal_requirements = dspy.OutputField()

    class GenerateDraftWortlautSignature(dspy.Signature):
        """
        Generate a draft of the 'Wortlaut' (exact wording) of the proposed constitutional amendment.
        """

        clarified_terms = dspy.InputField()
        legal_requirements = dspy.InputField()
        summarized_im_detail = dspy.InputField()
        arguments_committee = dspy.InputField()

        draft_wortlaut = dspy.OutputField()

    class LegalReviewSignature(dspy.Signature):
        """
        Review the draft 'Wortlaut' for legal accuracy and compliance, and produce the final 'Wortlaut'.
        """

        draft_wortlaut = dspy.InputField()

        final_wortlaut = dspy.OutputField()

    def __init__(self):
        super().__init__()
        #### Definition of modules ####
        self.summarize_im_detail_module = dspy.TypedPredictor(
            self.SummarizeImDetailSignature,
        )

        self.extract_arguments_committee_module = dspy.TypedPredictor(
            self.ExtractArgumentsCommitteeSignature,
        )

        self.extract_arguments_federal_council_module = dspy.TypedPredictor(
            self.ExtractArgumentsFederalCouncilSignature,
        )

        self.clarify_ambiguous_terms_module = dspy.TypedChainOfThought(
            self.ClarifyAmbiguousTermsSignature,
        )

        self.highlight_legal_requirements_module = dspy.TypedChainOfThought(
            self.HighlightLegalRequirementsSignature,
        )

        self.generate_draft_wortlaut_module = dspy.TypedChainOfThought(
            self.GenerateDraftWortlautSignature,
        )

        self.legal_review_module = dspy.TypedChainOfThought(
            self.LegalReviewSignature,
        )

    def get_draft_wortlaut_prediction(
        self,
        titel,
        im_detail,
        argumenteKomitee,
        empfehlungKomitee,
        argumenteBundesrat,
        empfehlungBundesrat,
    ):
        # Node A1: Summarize 'im_detail' (Predictor)
        summarized_im_detail = self.summarize_im_detail_module(
            titel=titel,
            im_detail=im_detail,
        ).summarized_im_detail

        # Node A2: Extract Arguments from Committee (Predictor)
        arguments_committee = self.extract_arguments_committee_module(
            titel=titel,
            argumenteKomitee=argumenteKomitee,
            empfehlungKomitee=empfehlungKomitee,
        ).arguments_committee

        # Node A3: Extract Arguments from Federal Council (Predictor)
        arguments_bundesrat = self.extract_arguments_federal_council_module(
            titel=titel,
            argumenteBundesrat=argumenteBundesrat,
            empfehlungBundesrat=empfehlungBundesrat,
        ).arguments_bundesrat

        # Node B1: Clarify Ambiguous Terms (ChainOfThought)
        clarified_terms = self.clarify_ambiguous_terms_module(
            summarized_im_detail=summarized_im_detail,
            arguments_committee=arguments_committee,
            arguments_bundesrat=arguments_bundesrat,
        ).clarified_terms

        # Node B2: Highlight Legal Requirements (ChainOfThought)
        legal_requirements = self.highlight_legal_requirements_module(
            summarized_im_detail=summarized_im_detail,
            clarified_terms=clarified_terms,
        ).legal_requirements

        # Bottleneck Node: Generate Draft 'Wortlaut' (ChainOfThought)
        draft_wortlaut = self.generate_draft_wortlaut_module(
            clarified_terms=clarified_terms,
            legal_requirements=legal_requirements,
            summarized_im_detail=summarized_im_detail,
            arguments_committee=arguments_committee,
        ).draft_wortlaut

        return draft_wortlaut

    def get_final_prediction(self, draft_wortlaut: str):
        prediction = self.legal_review_module(
            draft_wortlaut=draft_wortlaut,
        )
        return prediction

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
            self.get_draft_wortlaut_prediction(
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
    network: Network
    data: Dataset
    optimizer: BootstrapFewShotWithRandomSearch

    def optimize(
        self,
    ):
        trainset = self.data.data["train"]
        valset = self.data.data["validation"]

        if self.params.training_set_limit is not None:
            trainset = trainset[: self.params.training_set_limit]

        if self.params.valid_set_limit is not None:
            valset = valset[: self.params.valid_set_limit]

        # @todo pass kwargs through hyperparams
        optimized_network = self.optimizer.compile(
            student=self.network,
            trainset=trainset,
            valset=valset,
        )

        # Saving the optimized model
        iso_date = datetime.today().strftime("%Y-%m-%d")
        file_path = f"models/{iso_date}_{self.params.training_run_name}"
        file_extension = ".json"

        # Check if file exists; if yes, append a character until filename doesn't exist
        while os.path.exists(file_path + file_extension):
            file_path += "_new"

        optimized_network.save(file_path + file_extension)
