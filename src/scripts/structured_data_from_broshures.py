from typing_extensions import Literal
from pydantic import BaseModel
from openai import OpenAI

# ---- I/O ----
prompt = r"""
Du bist ein hilfreicher Chatbot der halb-strukturierte OCR-Texte in eine vorgeschriebene JSON-Struktur umwandelt.
Ich gebe dir den Text einer schweizerischen Abstimmungsbroschüre. Der Text wurde mit OCR extrahiert und ist somit nicht in einem 100% korrekten Format. Die Abstimmungsbroschüre beschreibt Schweizer Volksinitiativen. Für jede beschriebene Initiative liefert die Broschüre detaillierte Informationen für die Volksinitiativen, Argumente seitens des Initiativkomitees, des Bundesrates und Parlaments. Ignorier alle Markdown-Links zu Bildern ("xx_img__xx.png")

Aufgrund des OCR-Verfahrens ist der Text nicht verlässlich strukturiert. Bitte hilf mir, dem Text eine JSON-Struktur zu geben. Die Struktur dargestellt mit zwei Beispiel Volksiniativen:

```json
{
  "volksiniativen": [
    // Kommentar: Muster für eine Volksinitiative 1
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
    },
    // Kommentar: Muster für eine Volksinitiative 2
    {
      "titel": "Kostenbremse-Initiative"
      // Kommentar: Weitere Felder hier ...
    }
  ]
}
```
"""

# load in ocr-content from a file
with open("data/processed/brochure_ocr/2021_06_13/2021_06_13.md", "r") as file:
    # load every line into a raw string as if you would do r"""
    ocr_text = file.read()

# ---- Schema of the structured data ----
class Argument(BaseModel):
    ueberschrift: str
    inhalt: str

class ImDetail(BaseModel):
    ueberschrift: str
    inhalt: str

class Volksiniative(BaseModel):
    titel: str
    im_detail: list[ImDetail]
    argumenteKomitee: list[Argument]
    empfehlungKomitee: Literal["ja", "nein"]
    argumenteBundesrat: list[Argument]
    empfehlungKomitee: Literal["ja", "nein"]

class StructuredData(BaseModel):
    volksiniativen: list[Volksiniative]


# ---- OpenAI API call ----
client = OpenAI()

# todo: give examples
completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    temperature=0.2,
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": ocr_text}
    ],
    response_format=StructuredData,
)

message = completion.choices[0].message
if not message.parsed:
    print(message.refusal)

# todo: save it in a better way
# save the response to a file
with open("response.json", "w") as f:
    f.write(message.json())
