from typing import Tuple, Optional, List, Union
from pathlib import Path
import os
import pickle


import dspy
from dspy.datasets import DataLoader
import pandas as pd
from pydantic import BaseModel
from openai import OpenAI


# %% Helper functions
def get_path_to_truncated_wortlaut_pickles() -> Tuple[Path, Path]:
    """
    This function works in both script and Jupyter notebook environments.
    It tries different methods to find the project root.

    Returns:
        Tuple with paths to the train and validation datasets
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

    train_path = path.joinpath("data", "interim", "truncated_wortlaut_train.pkl")

    valid_path = path.joinpath("data", "interim", "truncated_wortlaut_valid.pkl")

    return (train_path, valid_path)


def get_path_to_bundesverfassung() -> Path:
    """
    This function works in both script and Jupyter notebook environments.

    Returns:
        Path to the Bundesverfassung plaintext file
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

    return path.joinpath("data", "processed", "bundesverfassung_de.md")


# %% Model definition


# Pydantic models for structured output
class ArticleNumber(BaseModel):
    number: str


class MatchingArticlesFromBundesverfassung(BaseModel):
    article_numbers: list[ArticleNumber]


class ArticleNumberRM(dspy.Retrieve):
    def __init__(self, k=3):
        """
        The 'k' parameter is not supported. Only kepy for compatibility with the dspy.Retrieve class.
        """
        assert k == 3, "setting 'k' parameter is not supported"
        super().__init__(k)
        self.client = OpenAI()
        self.bundesverfassung_text = ""
        with open(get_path_to_bundesverfassung(), "r") as f:
            self.bundesverfassung_text = f.read()

        if self.bundesverfassung_text == "":
            raise FileNotFoundError("Could not read Bundesverfassung text file")

    def get_prompt_for_query(self, query: str):
        # ---- preparing the prompt ----

        # create pseudo-xml tags around bundesverfassung
        opening_tag_bundesverfassung = "<bundesverfassung>\n"
        closing_tag_bundesverfassung = "\n</bundesverfassung>\n"

        bundesverfassung_part = (
            opening_tag_bundesverfassung
            + self.bundesverfassung_text
            + closing_tag_bundesverfassung
        )

        description = r"""
        <task-description>
        I just gave you the Swiss Federal Constitution (Bundesverfasssung in german, the language of the text). I am working on drafting new laws for the Bundesverfassung. The text of the drafts is already done, I want to find the relevant article in the Bundesverfassung which deals with the topics described in the draft.

        The draft has marked out sections that look like: Art. [ARTICLE NUMBER HERE]. For each of these sections, I want you to find and return corresponding article number in the Bundesverfassung. I just want a list of the matching Article numbers, chronologically ordered. Make sure you that if there are multiple sections, you always return the matching number of article numbers.
        </task-description>
        """

        opening_tag_draft = "<draft>\n"
        closing_tag_draft = "\n</draft>"

        draft_text = opening_tag_draft + query + closing_tag_draft
        prompt = bundesverfassung_part + description + draft_text

        return prompt

    def forward(
        self,
        query_or_queries: Union[str, List[str]],
        k: Optional[int] = None,
        by_prob: bool = False,
        with_metadata: bool = False,
        **kwargs,
    ) -> dspy.Prediction:
        """
        Everything but the query_or_queries argument is ignored.
        The parameters are kept for compatibility with the dspy.Retrieve class.
        """
        # ---- input validation ----
        assert k is None, "k parameter is not supported"
        assert not by_prob, "by_prob parameter is not supported"
        assert not with_metadata, "with_metadata parameter is not supported"
        assert not kwargs, "Additional keyword arguments are not supported"

        query = (
            " ".join(query_or_queries)
            if isinstance(query_or_queries, list)
            else query_or_queries
        )

        prompt = self.get_prompt_for_query(query)

        # ---- OpenAI API call ----
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            temperature=0.2,
            messages=[
                {"role": "system", "content": prompt},
            ],
            response_format=MatchingArticlesFromBundesverfassung,
        )

        message = completion.choices[0].message

        if not message.parsed:
            raise ValueError("message was not parsed")
        retrieved_article_numbers = [
            article.number for article in message.parsed.article_numbers
        ]

        # Ensure retrieved_article_numbers are always only digits
        retrieved_article_numbers = [
            "".join(filter(str.isdigit, num)) for num in retrieved_article_numbers
        ]

        return dspy.Prediction(retrieved_article_numbers=retrieved_article_numbers)
