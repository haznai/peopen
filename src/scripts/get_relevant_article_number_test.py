from pydantic import BaseModel
from openai import OpenAI


# ---- Prompt ----
# load in bundesverfassung data from data/processed/bundesverfassung_from_github.md
bundesverfassung_text = open(
    "../../data/processed/bundesverfassung_from_github.md", "r"
).read()


# create pseudo-xml tags around bundesverfassung
opening_tag_bundesverfassung = "<bundesverfassung>\n"
closing_tag_bundesverfassung = "\n</bundesverfassung>\n"

bundesverfassung_text = (
    opening_tag_bundesverfassung + bundesverfassung_text + closing_tag_bundesverfassung
)


# create prompt

description = r"""
<task-description>
I just gave you the Swiss Federal Constitution (Bundesverfasssung in german, the language of the text). I am working on drafting new laws for the Bundesverfassung. The text of the drafts is already done, I want to find the relevant article in the Bundesverfassung which deals with the topics described in the draft.

The draft has marked out sections that look like: Art. [ARTICLE NUMBER HERE]. For each of these sections, I want you to find and return corresponding article number in the Bundesverfassung. I just want a list of the matching Article numbers, chronologically ordered.
</task-description>
"""


opening_tag_draft = "<draft>\n"
closing_tag_draft = "\n</draft>"

draft_text = r"""
Die Bundesverfassung1 wird wie folgt geändert:
Art. [ARTICLE NUMBER HERE]
3 Versicherte haben Anspruch auf eine Verbilligung der Krankenversicherungsprämien. Die von den Versicherten zu übernehmenden Prämien betragen höchstens zehn Prozent des verfügbaren Einkommens. Die Prämienverbilligung wird zu mindestens zwei Dritteln durch den Bund und im verbleibenden Betrag durch die Kantone finanziert.
Art. [ARTICLE NUMBER HERE]
12. Übergangsbestimmung zu Art. 117 Abs. 3 (Verbilligung der Krankenversicherungsprämien)
Ist die Ausführungsgesetzgebung zu Artikel 117 Absatz 3 drei Jahre nach dessen Annahme durch Volk und Stände noch nicht in Kraft getreten, so erlässt der Bundesrat auf diesen Zeitpunkt hin die Ausführungsbestimmungen vorübergehend auf dem Verordnungsweg.
1 SR 101
2 Die endgültige Nummerierung dieses Absatzes wird nach der Volksabstimmung von der Bundeskanzlei festgelegt; dabei stimmt diese die Nummerierung ab auf die anderen geltenden Bestimmungen der Bundesverfassung und nimmt, wenn eine Anpassung der Nummerierung nötig ist, diese im ganzen Text der Initiative vor.
3 Die endgültige Ziffer dieser Übergangsbestimmung wird nach der Volksabstimmung von der Bundeskanzlei festgelegt
"""

draft_text = opening_tag_draft + draft_text + closing_tag_draft


prompt = bundesverfassung_text + description + draft_text


class ArticleNumber(BaseModel):
    number: str


class MatchingArticlesFromBundesverfassung(BaseModel):
    articles: list[ArticleNumber]


# ---- OpenAI API call ----
client = OpenAI()
completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    temperature=0.2,
    messages=[
        {"role": "system", "content": prompt},
    ],
    response_format=ArticleNumber,
)

message = completion.choices[0].message

if not message.parsed:
    raise ValueError("message was not parsed")

print(message)
