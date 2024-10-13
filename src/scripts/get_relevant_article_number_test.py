from pydantic import BaseModel
from openai import OpenAI
from tabulate import tabulate
import csv
import pickle


# %% ---- loading in the data ----
# load in bundesverfassung data from data/processed/bundesverfassung_from_github.md
bundesverfassung_text = open("../../data/processed/bundesverfassung_de.md", "r").read()


# %% ---- load in valid data of truncated wortlaut
# load in all the blacked_out_drafts and target article numbers
# data/processed/blackout_drafts_and_target_article_numbers.csv


# do preprocessing and sanity checks for the loaded in data
# final data should be a list of dictionaries with keys "blacked_out_paragraphs" and "target_article_numbers"
# in the `row` array
with open("../../data/processed/truncated_wortlaut_train.pickle", "rb") as f:
    pickle_data = pickle.load(f)

data_from_pickle = []

for i in range(len(pickle_data)):
    data_from_pickle.append(
        {
            "blacked_out_paragraphs": pickle_data[i]["wortlaut_truncated"],
            "target_article_numbers": pickle_data[i]["article_numbers"],
        }
    )


# Process and validate the data
for row in data_from_pickle:
    blacked_out_count = row["blacked_out_paragraphs"].count("[ARTICLE NUMBER HERE]")
    target_article_count = len(row["target_article_numbers"])

    if blacked_out_count != target_article_count:
        print(f"Warning: Mismatch in counts for row: {row}")
        raise ValueError("Mismatch in counts.")


# %% ---- Pydantic models ----
class ArticleNumber(BaseModel):
    number: str


class MatchingArticlesFromBundesverfassung(BaseModel):
    article_numbers: list[ArticleNumber]


# %% ---- preparing the prompt ----

# create pseudo-xml tags around bundesverfassung
opening_tag_bundesverfassung = "<bundesverfassung>\n"
closing_tag_bundesverfassung = "\n</bundesverfassung>\n"

bundesverfassung_text = (
    opening_tag_bundesverfassung + bundesverfassung_text + closing_tag_bundesverfassung
)


description = r"""
<task-description>
I just gave you the Swiss Federal Constitution (Bundesverfasssung in german, the language of the text). I am working on drafting new laws for the Bundesverfassung. The text of the drafts is already done, I want to find the relevant article in the Bundesverfassung which deals with the topics described in the draft.

The draft has marked out sections that look like: Art. [ARTICLE NUMBER HERE]. For each of these sections, I want you to find and return corresponding article number in the Bundesverfassung. I just want a list of the matching Article numbers, chronologically ordered. Make sure you that if there are multiple sections, you return the matching number of article numbers.
</task-description>
"""


opening_tag_draft = "<draft>\n"
closing_tag_draft = "\n</draft>"


# %% --- calling the model for each row in blackout_drafts_and_target_article_numbers ---

for row in data_from_pickle:
    draft_text = row["blacked_out_paragraphs"]
    draft_text = opening_tag_draft + draft_text + closing_tag_draft
    prompt = bundesverfassung_text + description + draft_text

    # %% ---- OpenAI API call ----
    client = OpenAI()
    completion = client.beta.chat.completions.parse(
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

    # print all the attributes of the parsed message

    # compare the parsed article numbers with the target article numbers
    # pretty print the results
    parsed_articles = [article.number for article in message.parsed.article_numbers]
    target_articles = row["target_article_numbers"]

    # Prepare data for the table
    table_data = []
    for parsed, target in zip(parsed_articles, target_articles):
        match = "✓" if parsed == target else "✗"
        table_data.append([f"Art. {parsed}", f"Art. {target}", match])

    # Create and print the table
    headers = ["Parsed Article", "Target Article", "Match"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    if len(parsed_articles) != len(target_articles):
        print(
            f"\nWarning: Number of parsed articles ({len(parsed_articles)}) "
            f"does not match number of target articles ({len(target_articles)})"
        )
