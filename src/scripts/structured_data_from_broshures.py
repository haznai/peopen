from typing_extensions import Literal
from pydantic import BaseModel
from openai import OpenAI
import os

# ---- I/O ----
# get all files ending in .md in the ocr'd broshures (recursively)

folder_path = 'data/processed/volksinitiatives'
md_file_paths = []

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.md'):
            md_file_paths.append(os.path.join(root, file))

# ----  Prompt and schema definition ----

prompt = r"""
Du bist ein hilfreicher Chatbot der halb-strukturierte OCR-Texte in eine vorgeschriebene JSON-Struktur umwandelt.
Ich gebe dir den Text einer schweizerischen Abstimmungsbroschüre. Der Text wurde mit OCR extrahiert und ist somit nicht in einem 100% korrekten Format. Die Abstimmungsbroschüre beschreibt Schweizer Volksabstimmungen. Für jede Volksabstimmung liefert die Broschüre detaillierte Informationen für die Volksinitiativen, Argumente seitens des Initiativkomitees, des Bundesrates und Parlaments.
Ich werde dir jeweils eine Volksabstimmung geben. Ignorier alle Markdown-Links zu Bildern ("xx_img__xx.png"), sie beinhalten keine relevanten Informationen.

Aufgrund des OCR-Verfahrens ist der Text nicht verlässlich strukturiert. Bitte hilf mir, dem Text eine JSON-Struktur zu geben. Die JSON-Struktur ist wie folgt aufgebaut:

```json
{
    "titel": "Prämien-Entlastungs-Initiative",
    "im_detail": [
    {
        "ueberschrift": "Ausgangslage",
        "inhalt": "Wer in der Schweiz krank ist, erhält die nötige medizinische Behandlung. Seit 1996 übernimmt die obligatorische Krankenversicherung die Kosten dafür. Die Krankenversicherung wird über die Krankenkassenprämien und Kostenbeteiligungen (Franchise, Selbstbehalt, Spitalkostenbeitrag) finanziert. Die Kosten der Krankenversicherung sind in den letzten Jahrzehnten stark gestiegen. Um sie zu decken, mussten die Prämien entsprechend erhöht werden. Die Prämien stiegen im Verhältnis deutlich mehr als die Löhne"
    },
    {
        "ueberschrift": "Prämienverbilligung",
        "inhalt": "Die Prämien werden pro Person und unabhängig von derEinkommenshöhe bestimmt. Die Kantone sind verpflichtet, die Prämien der Versicherten in bescheidenen wirtschaftlichen Verhältnissen zu verbilligen. Die Kantone erhalten dazu vomBund einen Beitrag. Der Mittelstand profitiert jedoch nicht oder nur teilweise von dieser Verbilligung und wird darum von den steigenden Prämien zunehmend stark belastet"
    },
    // Kommentar: Weitere Paragraphen hier ...
    ],
    "argumenteKomitee": [
    {
        "ueberschrift": "Worum geht es?",
        "inhalt": "Die Krankenkassenprämien steigen seit Jahren und reissen ein immer grösseres Loch in unser Portemonnaie. Bis zu 15 000 Franken: So viel zahlt heute eine vierköpfige Familie pro Jahr für die Krankenkasse. Die Prämienexplosion ist aber nur ein Spiegelbild der steigenden Kosten im Gesundheitswesen. Um das Problem nachhaltig zu lösen, braucht es jetzt die Kostenbremse."
    },
    {
        "ueberschrift": "Drohen Rationierungen?",
        "inhalt": "Nein. Im Gegenteil: Die Initiative will, dass alle Akteur eendlich Verantwortung für die Kostenexplosion übernehmen und der interne Verteilkampf zulasten der Prämienzahlenden aufhört. Während Hausärztinnen, Kinderärzte und Pflegende schon heute die Lasten des Systems tragen, bereichern sich andere schamlos."
    }
    // Kommentar: Weitere Argumente hier ...
    ],
    "empfehlungKomitee": "ja",
    "argumenteBundesrat": [
    {
        "ueberschrift": "Richtige Diagnose, falsches Mittel",
        "inhalt": "Die Initiative greift ein wichtiges Problem auf: Die Kosten in der obligatorischen Krankenversicherung steigen zu stark. Es gibt ineffiziente Strukturen und es werden mehr Behandlungen durchgeführt, als medizinisch nötig wären. Die Initiative ist aber zu starr: Sie bindet das erlaubte Kostenwachstum einseitig an die Entwicklung der Löhne und der Wirtschaft. Damit werden nachvollziehbare Gründe für das Kostenwachstum ausgeblendet, beispielsweise der medizinische Fortschritt oder die Alterung der Bevölkerung."
    }
    // Kommentar: Weitere Argumente hier ...
    ],
    "empfehlungBundesrat": "nein"
}
```

Stelle sicher, dass du kein Argument oder Detail auslässt.
"""

class Argument(BaseModel):
    ueberschrift: str
    inhalt: str

class ImDetail(BaseModel):
    ueberschrift: str
    inhalt: str

class Volksabstimmung(BaseModel):
    titel: str
    im_detail: list[ImDetail]
    argumenteKomitee: list[Argument]
    empfehlungKomitee: Literal["ja", "nein"]
    argumenteBundesrat: list[Argument]
    empfehlungBundesrat: Literal["ja", "nein"]



# ---- OpenAI API call ----
client = OpenAI()

for path in md_file_paths:
    # read ocr"d content
    with open(path, "r") as file:
        ocr_text = file.read()

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        temperature=0.2,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": ocr_text}
        ],
        response_format=Volksabstimmung
    )

    message = completion.choices[0].message

    if not message.parsed:
        raise ValueError(f"{path} was not parsed")

    # Extract date and initiative number from the original path
    date_folder = os.path.basename(os.path.dirname(path))
    initiative_file = os.path.basename(path)

    # Create the new directory structure
    new_dir = os.path.join('data', 'volksabstimmung_structured', date_folder)
    os.makedirs(new_dir, exist_ok=True)

    # Create the new file path
    new_file_path = os.path.join(new_dir, initiative_file)

    # Save the structured data to the new file
    with open(new_file_path, 'w') as file:
        file.write(message.model_dump_json(indent=2))

    print(f"Structured data saved to: {new_file_path}")
