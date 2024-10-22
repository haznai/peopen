from dataclasses import dataclass
import dspy
import transformers
from transformers import TextGenerationPipeline
from pathlib import Path
import torch
import os


# ---- Helper Functions ----
def load_model() -> TextGenerationPipeline:
    """
    This function works in both script and Jupyter notebook environments.
    It tries different methods to find the safetensors of the factual-consistency model in the `models` directory.

    Returns:
        TextGenerationPipeline: The transformers pipeline object for the factual-consistency model.
    """
    try:
        # Try to get the path of the current file (works in scripts)
        path = Path(__file__).resolve().parent.parent.parent
    except NameError:
        try:
            # Try to get the path from Jupyter notebook
            import IPython

            path = Path(IPython.get_ipython().kernel.profile_dir).parent.parent.parent  # type: ignore
        except Exception:
            # Fallback to current working directory
            path = Path(os.getcwd()).parent.parent.resolve()

    model_path = path.joinpath("models", "ragarwal__factual-consistency-llama3-8b")

    pipeline = transformers.pipeline(
        "text-generation",
        model=str(model_path),
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="mps",
    )

    return pipeline  # type: ignore


# ---- Define the Metrics and Model ----
@dataclass(frozen=True)
class FactualConsistencyMetrics:
    """
    Defines calculation of evaluation metrics by the factual consistency model.
    """

    model: TextGenerationPipeline

    def is_yes_or_no(self, assistant_answer: str | None) -> bool:
        """
        The factual consistency model answers with a "Yes" or "No" to the question of whether the target text can be inferred from the source text.

        We check if the assistant's answer is a "Yes" or "No" to determine if the model's response is valid.
        """
        if assistant_answer is None:
            return False

        return "yes" in assistant_answer.lower()

    def is_factually_consistent(self, pred: str, source: str) -> bool:
        PROMPT = """Can the target text be inferred from the source text? In other words, is the target text factually consistent with the source text? Answer with a "Yes" or "No"."""

        answer = self.model(
            [
                {
                    "role": "user",
                    "content": f"{PROMPT}\n\n_target_ {pred}\n\n_source_ {source}",
                }
            ],
            max_length=4096,
            clean_up_tokenization_spaces=True,
        )

        # Initialize a variable to hold the assistant's content
        assistant_content = None

        # Navigate through the structure to find the assistant's message
        for item in answer:  # type: ignore
            generated_text = item.get("generated_text", [])  # type: ignore
            for message in generated_text:
                if message.get("role") == "assistant":
                    assistant_content = message.get("content")
                    break  # Exit the inner loop once found
            if assistant_content:
                break  # Exit the outer loop if assistant's content is found

        return self.is_yes_or_no(assistant_content)

    def get_score(self, example, prediction, trace=None):
        # function signature is important to be compatible with dspy optimizers

        # get the predicted values (summarized)
        pred_im_detail = prediction.summarized_im_detail
        pred_arguments_committee = prediction.arguments_committee
        pred_arguments_bundesrat = prediction.arguments_bundesrat

        # get the original values (unsummarized)
        im_detail = example.im_detail
        argumenteKomitee = example.argumenteKomitee
        argumenteBundesrat = example.argumenteBundesrat

        list_preds = [
            pred_im_detail,
            pred_arguments_committee,
            pred_arguments_bundesrat,
        ]
        list_sources = [im_detail, argumenteKomitee, argumenteBundesrat]

        score = 0
        for pred, source in zip(list_preds, list_sources):
            if self.is_factually_consistent(pred, source):
                score += 1

        return score / len(list_preds)


class FactualConsistencyNetwork(dspy.Module):
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

    def get_first_summarizations(
        self,
        titel,
        im_detail,
        argumenteKomitee,
        empfehlungKomitee,
        argumenteBundesrat,
        empfehlungBundesrat,
    ):
        summarized_im_detail = self.summarize_im_detail_module(
            titel=titel,
            im_detail=im_detail,
        ).summarized_im_detail

        arguments_committee = self.extract_arguments_committee_module(
            titel=titel,
            argumenteKomitee=argumenteKomitee,
            empfehlungKomitee=empfehlungKomitee,
        ).arguments_committee

        arguments_bundesrat = self.extract_arguments_federal_council_module(
            titel=titel,
            argumenteBundesrat=argumenteBundesrat,
            empfehlungBundesrat=empfehlungBundesrat,
        ).arguments_bundesrat

        return dspy.Prediction(
            summarized_im_detail=summarized_im_detail,
            arguments_committee=arguments_committee,
            arguments_bundesrat=arguments_bundesrat,
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
        return self.get_first_summarizations(
            titel,
            im_detail,
            argumenteKomitee,
            empfehlungKomitee,
            argumenteBundesrat,
            empfehlungBundesrat,
        )
