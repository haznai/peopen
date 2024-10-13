import pickle
from pydantic import BaseModel
from openai import OpenAI


# %% hand label `train` dataset
with open(
    "../../data/processed/volksiniativen_with_wortlaut_dspy_dataset_train.pkl", "rb"
) as file:
    data_train = pickle.load(file)


# --- hand labeled valid dataset (confirmed to be correct) ----
valid_1 = [
    r"""Die Bundesverfassung1 wird wie folgt geändert:
Art. 117 Abs. 32
3 Versicherte haben Anspruch auf eine Verbilligung der Krankenversicherungsprämien. Die von den Versicherten zu übernehmenden Prämien betragen höchstens zehn Prozent des verfügbaren Einkommens. Die Prämienverbilligung wird zu mindestens zwei Dritteln durch den Bund und im verbleibenden Betrag durch die Kantone finanziert.
Art. 197 Ziff. 123
12. Übergangsbestimmung zu Art. 117 Abs. 3 (Verbilligung der Krankenversicherungsprämien)
Ist die Ausführungsgesetzgebung zu Artikel 117 Absatz 3 drei Jahre nach dessen Annahme durch Volk und Stände noch nicht in Kraft getreten, so erlässt der Bundesrat auf diesen Zeitpunkt hin die Ausführungsbestimmungen vorübergehend auf dem Verordnungsweg.
1 SR 101
2 Die endgültige Nummerierung dieses Absatzes wird nach der Volksabstimmung von der Bundeskanzlei festgelegt; dabei stimmt diese die Nummerierung ab auf die anderen geltenden Bestimmungen der Bundesverfassung und nimmt, wenn eine Anpassung der Nummerierung nötig ist, diese im ganzen Text der Initiative vor.
3 Die endgültige Ziffer dieser Übergangsbestimmung wird nach der Volksabstimmung von der Bundeskanzlei festgelegt""",
    r"""
Art. [ARTICLE NUMBER HERE]
3 Versicherte haben Anspruch auf eine Verbilligung der Krankenversicherungsprämien. Die von den Versicherten zu übernehmenden Prämien betragen höchstens zehn Prozent des verfügbaren Einkommens. Die Prämienverbilligung wird zu mindestens zwei Dritteln durch den Bund und im verbleibenden Betrag durch die Kantone finanziert.

Art. [ARTICLE NUMBER HERE]
12. Übergangsbestimmung zu Art. 117 Abs. 3 (Verbilligung der Krankenversicherungsprämien)
Ist die Ausführungsgesetzgebung zu Artikel 117 Absatz 3 drei Jahre nach dessen Annahme durch Volk und Stände noch nicht in Kraft getreten, so erlässt der Bundesrat auf diesen Zeitpunkt hin die Ausführungsbestimmungen vorübergehend auf dem Verordnungsweg.
""",
    ["117", "197"],
]

valid_2 = [
    r"""Die Bundesverfassung1  wird wie folgt geändert:
Art. 112 Abs. 2 Bst. ater
2 Er [der Bund] beachtet dabei [beim Erlass der Vorschriften über die Alters-, Hinterlassenen- und Invalidenvorsorge] folgende Grundsätze:
ater.  Das Rentenalter ist an die durchschnittliche Lebenserwartung der schweizerischen Wohnbevölkerung im Alter von 65 Jahren gebunden; diese Lebenserwartung am 1. Januar des vierten Jahres nach Inkrafttreten dieser Bestimmung wird als Referenzwert festgesetzt; das Rentenalter entspricht der Differenz zwischen der Lebenserwartung und dem Referenzwert, multipliziert mit dem Faktor 0,8 zuzüglich 66; die Anpassung des Rentenalters erfolgt jährlich in Schritten von höchstens zwei Monaten; das Rentenalter wird den betroffenen Personen fünf Jahre vor Erreichen des Rentenalters bekannt gegeben;
Art. 197 Ziff. 122 
12. Übergangsbestimmung zu Art. 112 Abs. 2 Bst. ater (Rentenalter)
1 Ab dem 1. Januar des vierten Jahres nach Annahme von Artikel 112 Absatz 2 Buchstabe ater wird das Rentenalter für Männer in Schritten von jeweils zwei Monaten pro Jahr erhöht, bis es 66 Jahre beträgt.
2 Ab dem 1. Januar des vierten Jahres nach Annahme von Artikel 112 Absatz 2 Buchstabe ater wird das Rentenalter für Frauen in Schritten von jeweils vier Monaten pro Jahr erhöht, bis es dem Rentenalter für Männer entspricht. Anschliessend wird das Rentenalter für Frauen in Schritten von jeweils zwei Monaten pro Jahr erhöht, bis es 66 Jahre beträgt.
3 Ab dem 1. Januar des vierten Jahres nach Annahme von Artikel 112 Absatz 2 Buchstabe ater wird das Rentenalter an die durchschnittliche Lebenserwartung der schweizerischen Wohnbevölkerung im Alter von 65 Jahren gebunden.
4 Sind die Ausführungsbestimmungen zu Artikel 112 Absatz 2 Buchstabe ater drei Jahre nach dessen Annahme noch nicht in Kraft getreten, erlässt der Bundesrat auf den 1. Januar des vierten auf die Annahme folgenden Jahres die erforderlichen Ausführungsbestimmungen durch Verordnung. Die Verordnung gilt bis zum Inkrafttreten der gesetzlichen Bestimmungen. Der Bundesrat kann in der Verordnung von der Gesetzgebung zur Alters- und Hinterlassenenversicherung abweichen.
1 SR 101
2 Die endgültige Ziffer dieser Übergangsbestimmungen wird nach der Volksabstimmung von der Bundeskanzlei festgelegt.
""",
    r""""
Art. [ARTICLE NUMBER HERE]
2 Er [der Bund] beachtet dabei [beim Erlass der Vorschriften über die Alters-, Hinterlassenen- und Invalidenvorsorge] folgende Grundsätze:
ater.  Das Rentenalter ist an die durchschnittliche Lebenserwartung der schweizerischen Wohnbevölkerung im Alter von 65 Jahren gebunden; diese Lebenserwartung am 1. Januar des vierten Jahres nach Inkrafttreten dieser Bestimmung wird als Referenzwert festgesetzt; das Rentenalter entspricht der Differenz zwischen der Lebenserwartung und dem Referenzwert, multipliziert mit dem Faktor 0,8 zuzüglich 66; die Anpassung des Rentenalters erfolgt jährlich in Schritten von höchstens zwei Monaten; das Rentenalter wird den betroffenen Personen fünf Jahre vor Erreichen des Rentenalters bekannt gegeben;
Art. [ARTICLE NUMBER HERE]
12. Übergangsbestimmung zu Art. 112 Abs. 2 Bst. ater (Rentenalter)
1 Ab dem 1. Januar des vierten Jahres nach Annahme von Artikel 112 Absatz 2 Buchstabe ater wird das Rentenalter für Männer in Schritten von jeweils zwei Monaten pro Jahr erhöht, bis es 66 Jahre beträgt.
2 Ab dem 1. Januar des vierten Jahres nach Annahme von Artikel 112 Absatz 2 Buchstabe ater wird das Rentenalter für Frauen in Schritten von jeweils vier Monaten pro Jahr erhöht, bis es dem Rentenalter für Männer entspricht. Anschliessend wird das Rentenalter für Frauen in Schritten von jeweils zwei Monaten pro Jahr erhöht, bis es 66 Jahre beträgt.
3 Ab dem 1. Januar des vierten Jahres nach Annahme von Artikel 112 Absatz 2 Buchstabe ater wird das Rentenalter an die durchschnittliche Lebenserwartung der schweizerischen Wohnbevölkerung im Alter von 65 Jahren gebunden.
4 Sind die Ausführungsbestimmungen zu Artikel 112 Absatz 2 Buchstabe ater drei Jahre nach dessen Annahme noch nicht in Kraft getreten, erlässt der Bundesrat auf den 1. Januar des vierten auf die Annahme folgenden Jahres die erforderlichen Ausführungsbestimmungen durch Verordnung. Die Verordnung gilt bis zum Inkrafttreten der gesetzlichen Bestimmungen. Der Bundesrat kann in der Verordnung von der Gesetzgebung zur Alters- und Hinterlassenenversicherung abweichen.
""",
    ["112", "197"],
]

valid_3 = [
    r"""
Die Bundesverfassung1 wird wie folgt geändert:
Art. 117 Abs. 3 und 4
3 Er [der Bund] regelt in Zusammenarbeit mit den Kantonen, den Krankenversicherern und den Leistungserbringern die Kostenübernahme durch die obligatorische Krankenpflegeversicherung so, dass sich mit wirksamen Anreizen die Kosten entsprechend der schweizerischen Gesamtwirtschaft und den durchschnittlichen Löhnen entwickeln. Er führt dazu eine Kostenbremse ein.
4 Das Gesetz regelt die Einzelheiten.
Art. 197 Ziff. 122
12. Übergangsbestimmung zu Art. 117 Abs. 3 und 4 (Kranken- und Unfallversicherung)
Liegt die Steigerung der durchschnittlichen Kosten je versicherte Person und Jahr in der obligatorischen Krankenpflegeversicherung zwei Jahre nach Annahme von Artikel 117 Absätze 3 und 4 durch Volk und Stände mehr als ein Fünftel über der Entwicklung der Nominallöhne und haben die Krankenversicherer und die Leistungserbringer (Tarifpartner) bis zu diesem Zeitpunkt keine verbindlichen Massnahmen zur Kostendämpfung festgelegt, so ergreift der Bund in Zusammenarbeit mit den Kantonen Massnahmen zur Kostensenkung, die ab dem nachfolgenden Jahr wirksam werden.
1 SR 101
2 Die endgültige Ziffer dieser Übergangsbestimmung wird nach der Volksabstimmung von der Bundeskanzlei festgelegt.""",
    r"""
Art. [ARTICLE NUMBER HERE]
3 Er [der Bund] regelt in Zusammenarbeit mit den Kantonen, den Krankenversicherern und den Leistungserbringern die Kostenübernahme durch die obligatorische Krankenpflegeversicherung so, dass sich mit wirksamen Anreizen die Kosten entsprechend der schweizerischen Gesamtwirtschaft und den durchschnittlichen Löhnen entwickeln. Er führt dazu eine Kostenbremse ein.
4 Das Gesetz regelt die Einzelheiten.
Art. [ARTICLE NUMBER HERE]
12. Übergangsbestimmung zu Art. 117 Abs. 3 und 4 (Kranken- und Unfallversicherung)
Liegt die Steigerung der durchschnittlichen Kosten je versicherte Person und Jahr in der obligatorischen Krankenpflegeversicherung zwei Jahre nach Annahme von Artikel 117 Absätze 3 und 4 durch Volk und Stände mehr als ein Fünftel über der Entwicklung der Nominallöhne und haben die Krankenversicherer und die Leistungserbringer (Tarifpartner) bis zu diesem Zeitpunkt keine verbindlichen Massnahmen zur Kostendämpfung festgelegt, so ergreift der Bund in Zusammenarbeit mit den Kantonen Massnahmen zur Kostensenkung, die ab dem nachfolgenden Jahr wirksam werden.
""",
    ["117", "197"],
]

valid_4 = [
    r"""
Die Bundesverfassung1 wird wie folgt geändert:
Art. 10 Abs. 2bis
2bis Eingriffe in die körperliche oder geistige Unversehrtheit einer Person bedürfen deren Zustimmung. Die betroffene Person darf aufgrund der Verweigerung der Zustimmung weder bestraft werden noch dürfen ihr soziale oder berufliche Nachteile erwachsen.
Art. 197 Ziff. 122
12. Übergangsbestimmung zu Art. 10 Abs. 2bis (Recht auf körperliche und geistige Unversehrtheit)
Die Bundesversammlung erlässt die Ausführungsbestimmungen zu Artikel 10 Absatz 2bis spätestens ein Jahr nach dessen Annahme durch Volk und Stände. Treten die Ausführungsbestimmungen innerhalb dieser Frist nicht in Kraft, so erlässt der Bundesrat die Ausführungsbestimmungen in Form einer Verordnung und setzt sie auf diesen Zeitpunkt hin in Kraft. Die Verordnung gilt bis zum Inkrafttreten der von der Bundesversammlung erlassenen Ausführungsbestimmungen.
1 SR 101
2 Die engültige Ziffer dieser Übergangsbestimmung wird nach der Volksabstimmung von der Bundeskanzlei festgelegt.
""",
    r"""
Art. [ARTICLE NUMBER HERE]
2bis Eingriffe in die körperliche oder geistige Unversehrtheit einer Person bedürfen deren Zustimmung. Die betroffene Person darf aufgrund der Verweigerung der Zustimmung weder bestraft werden noch dürfen ihr soziale oder berufliche Nachteile erwachsen.
Art. [ARTICLE NUMBER HERE]
12. Übergangsbestimmung zu Art. 10 Abs. 2bis (Recht auf körperliche und geistige Unversehrtheit)
Die Bundesversammlung erlässt die Ausführungsbestimmungen zu Artikel 10 Absatz 2bis spätestens ein Jahr nach dessen Annahme durch Volk und Stände. Treten die Ausführungsbestimmungen innerhalb dieser Frist nicht in Kraft, so erlässt der Bundesrat die Ausführungsbestimmungen in Form einer Verordnung und setzt sie auf diesen Zeitpunkt hin in Kraft. Die Verordnung gilt bis zum Inkrafttreten der von der Bundesversammlung erlassenen Ausführungsbestimmungen.
""",
    ["10", "197"],
]


valid_5 = [
    r"""
Die Bundesverfassung1 wird wie folgt geändert:
Art. 197 Ziff. 122
12. Übergangsbestimmung zu Art. 112 (Alters-, Hinterlassenen- und Invalidenversicherung)
1 Bezügerinnen und Bezüger einer Altersrente haben Anspruch auf einen jährlichen Zuschlag in der Höhe eines Zwölftels ihrer jährlichen Rente.
2 Der Anspruch auf den jährlichen Zuschlag entsteht spätestens mit Beginn des zweiten Kalenderjahres, das der Annahme dieser Bestimmung durch Volk und Stände folgt.
3 Das Gesetz stellt sicher, dass der jährliche Zuschlag weder zu einer Reduktion der Ergänzungsleistungen noch zum Verlust des Anspruchs auf diese Leistungen führt.
1 SR 101
2 Die endgültige Ziffer dieser Übergangsbestimmung wird nach der Volksabstimmung von der Bundeskanzlei festgelegt.
""",
    r"""
Art. [ARTICLE NUMBER HERE]
12. Übergangsbestimmung zu Art. 112 (Alters-, Hinterlassenen- und Invalidenversicherung)
1 Bezügerinnen und Bezüger einer Altersrente haben Anspruch auf einen jährlichen Zuschlag in der Höhe eines Zwölftels ihrer jährlichen Rente.
2 Der Anspruch auf den jährlichen Zuschlag entsteht spätestens mit Beginn des zweiten Kalenderjahres, das der Annahme dieser Bestimmung durch Volk und Stände folgt.
3 Das Gesetz stellt sicher, dass der jährliche Zuschlag weder zu einer Reduktion der Ergänzungsleistungen noch zum Verlust des Anspruchs auf diese Leistungen führt.
""",
    ["197"],
]


# %% Prepare data
class ArticleNumber(BaseModel):
    number: str


class ModifiedLawText(BaseModel):
    processed_text: str
    extracted_article_numbers: list[ArticleNumber]


# %% Prepare prompt

prompt = r"""

[Task 1 Description]

Please process the following Swiss law drafts by removing the standard introductory text at the beginning and any footnotes or concluding remarks at the end. Focus on retaining the main content of the articles EXACTLY while just removing the footnotes after `1 SR 101` (or similiar).

[/Task 1 Description]

[Task 2 Description]

Extract the article numbers from the text and replace them with "[ARTICLE NUMBER HERE]" in the main body of the text. Collect the extracted article numbers in a separate list. Treat these tasks as sequential steps in your processing. Make sure to remove extra information after the "[ARTICLE NUMBER HERE]" placeholder aswell, like "Abs. X", "Bst. X", "Ziff. X", etc. I will give you examples, just follow the same pattern.

[/Task 2 Description]

[Examples]

**Example 1:**

Original Text:

```plaintext
Die Bundesverfassung1 wird wie folgt geändert:
Art. 117 Abs. 32
3 Versicherte haben Anspruch auf eine Verbilligung der Krankenversicherungsprämien. Die von den Versicherten zu übernehmenden Prämien betragen höchstens zehn Prozent des verfügbaren Einkommens. Die Prämienverbilligung wird zu mindestens zwei Dritteln durch den Bund und im verbleibenden Betrag durch die Kantone finanziert.
Art. 197 Ziff. 123
12. Übergangsbestimmung zu Art. 117 Abs. 3 (Verbilligung der Krankenversicherungsprämien)
Ist die Ausführungsgesetzgebung zu Artikel 117 Absatz 3 drei Jahre nach dessen Annahme durch Volk und Stände noch nicht in Kraft getreten, so erlässt der Bundesrat auf diesen Zeitpunkt hin die Ausführungsbestimmungen vorübergehend auf dem Verordnungsweg.
1 SR 101
2 Die endgültige Nummerierung dieses Absatzes wird nach der Volksabstimmung von der Bundeskanzlei festgelegt; dabei stimmt diese die Nummerierung ab auf die anderen geltenden Bestimmungen der Bundesverfassung und nimmt, wenn eine Anpassung der Nummerierung nötig ist, diese im ganzen Text der Initiative vor.
3 Die endgültige Ziffer dieser Übergangsbestimmung wird nach der Volksabstimmung von der Bundeskanzlei festgelegt
```

Processed Text:

```plaintext
Art. [ARTICLE NUMBER HERE]
3 Versicherte haben Anspruch auf eine Verbilligung der Krankenversicherungsprämien. Die von den Versicherten zu übernehmenden Prämien betragen höchstens zehn Prozent des verfügbaren Einkommens. Die Prämienverbilligung wird zu mindestens zwei Dritteln durch den Bund und im verbleibenden Betrag durch die Kantone finanziert.

Art. [ARTICLE NUMBER HERE]
12. Übergangsbestimmung zu Art. 117 Abs. 3 (Verbilligung der Krankenversicherungsprämien)
Ist die Ausführungsgesetzgebung zu Artikel 117 Absatz 3 drei Jahre nach dessen Annahme durch Volk und Stände noch nicht in Kraft getreten, so erlässt der Bundesrat auf diesen Zeitpunkt hin die Ausführungsbestimmungen vorübergehend auf dem Verordnungsweg.
```

Extracted Article Numbers:

```plaintext
["117", "197"]
```

---

**Example 2:**

Original Text:

```plaintext
Die Bundesverfassung1  wird wie folgt geändert:
Art. 112 Abs. 2 Bst. ater
2 Er [der Bund] beachtet dabei [beim Erlass der Vorschriften über die Alters-, Hinterlassenen- und Invalidenvorsorge] folgende Grundsätze:
ater.  Das Rentenalter ist an die durchschnittliche Lebenserwartung der schweizerischen Wohnbevölkerung im Alter von 65 Jahren gebunden; diese Lebenserwartung am 1. Januar des vierten Jahres nach Inkrafttreten dieser Bestimmung wird als Referenzwert festgesetzt; das Rentenalter entspricht der Differenz zwischen der Lebenserwartung und dem Referenzwert, multipliziert mit dem Faktor 0,8 zuzüglich 66; die Anpassung des Rentenalters erfolgt jährlich in Schritten von höchstens zwei Monaten; das Rentenalter wird den betroffenen Personen fünf Jahre vor Erreichen des Rentenalters bekannt gegeben;
Art. 197 Ziff. 122 
12. Übergangsbestimmung zu Art. 112 Abs. 2 Bst. ater (Rentenalter)
1 Ab dem 1. Januar des vierten Jahres nach Annahme von Artikel 112 Absatz 2 Buchstabe ater wird das Rentenalter für Männer in Schritten von jeweils zwei Monaten pro Jahr erhöht, bis es 66 Jahre beträgt.
2 Ab dem 1. Januar des vierten Jahres nach Annahme von Artikel 112 Absatz 2 Buchstabe ater wird das Rentenalter für Frauen in Schritten von jeweils vier Monaten pro Jahr erhöht, bis es dem Rentenalter für Männer entspricht. Anschliessend wird das Rentenalter für Frauen in Schritten von jeweils zwei Monaten pro Jahr erhöht, bis es 66 Jahre beträgt.
3 Ab dem 1. Januar des vierten Jahres nach Annahme von Artikel 112 Absatz 2 Buchstabe ater wird das Rentenalter an die durchschnittliche Lebenserwartung der schweizerischen Wohnbevölkerung im Alter von 65 Jahren gebunden.
4 Sind die Ausführungsbestimmungen zu Artikel 112 Absatz 2 Buchstabe ater drei Jahre nach dessen Annahme noch nicht in Kraft getreten, erlässt der Bundesrat auf den 1. Januar des vierten auf die Annahme folgenden Jahres die erforderlichen Ausführungsbestimmungen durch Verordnung. Die Verordnung gilt bis zum Inkrafttreten der gesetzlichen Bestimmungen. Der Bundesrat kann in der Verordnung von der Gesetzgebung zur Alters- und Hinterlassenenversicherung abweichen.
1 SR 101
2 Die endgültige Ziffer dieser Übergangsbestimmungen wird nach der Volksabstimmung von der Bundeskanzlei festgelegt.
```

Processed Text:

```plaintext
Art. [ARTICLE NUMBER HERE]
2 Er [der Bund] beachtet dabei [beim Erlass der Vorschriften über die Alters-, Hinterlassenen- und Invalidenvorsorge] folgende Grundsätze:
ater.  Das Rentenalter ist an die durchschnittliche Lebenserwartung der schweizerischen Wohnbevölkerung im Alter von 65 Jahren gebunden; diese Lebenserwartung am 1. Januar des vierten Jahres nach Inkrafttreten dieser Bestimmung wird als Referenzwert festgesetzt; das Rentenalter entspricht der Differenz zwischen der Lebenserwartung und dem Referenzwert, multipliziert mit dem Faktor 0,8 zuzüglich 66; die Anpassung des Rentenalters erfolgt jährlich in Schritten von höchstens zwei Monaten; das Rentenalter wird den betroffenen Personen fünf Jahre vor Erreichen des Rentenalters bekannt gegeben;

Art. [ARTICLE NUMBER HERE]
12. Übergangsbestimmung zu Art. 112 Abs. 2 Bst. ater (Rentenalter)
1 Ab dem 1. Januar des vierten Jahres nach Annahme von Artikel 112 Absatz 2 Buchstabe ater wird das Rentenalter für Männer in Schritten von jeweils zwei Monaten pro Jahr erhöht, bis es 66 Jahre beträgt.
2 Ab dem 1. Januar des vierten Jahres nach Annahme von Artikel 112 Absatz 2 Buchstabe ater wird das Rentenalter für Frauen in Schritten von jeweils vier Monaten pro Jahr erhöht, bis es dem Rentenalter für Männer entspricht. Anschliessend wird das Rentenalter für Frauen in Schritten von jeweils zwei Monaten pro Jahr erhöht, bis es 66 Jahre beträgt.
3 Ab dem 1. Januar des vierten Jahres nach Annahme von Artikel 112 Absatz 2 Buchstabe ater wird das Rentenalter an die durchschnittliche Lebenserwartung der schweizerischen Wohnbevölkerung im Alter von 65 Jahren gebunden.
4 Sind die Ausführungsbestimmungen zu Artikel 112 Absatz 2 Buchstabe ater drei Jahre nach dessen Annahme noch nicht in Kraft getreten, erlässt der Bundesrat auf den 1. Januar des vierten auf die Annahme folgenden Jahres die erforderlichen Ausführungsbestimmungen durch Verordnung. Die Verordnung gilt bis zum Inkrafttreten der gesetzlichen Bestimmungen. Der Bundesrat kann in der Verordnung von der Gesetzgebung zur Alters- und Hinterlassenenversicherung abweichen.
```

Extracted Article Numbers:

```plaintext
["112", "197"]
```

---

**Example 3:**

Original Text:

```plaintext
Die Bundesverfassung1 wird wie folgt geändert:
Art. 197 Ziff. 122
12. Übergangsbestimmung zu Art. 112 (Alters-, Hinterlassenen- und Invalidenversicherung)
1 Bezügerinnen und Bezüger einer Altersrente haben Anspruch auf einen jährlichen Zuschlag in der Höhe eines Zwölftels ihrer jährlichen Rente.
2 Der Anspruch auf den jährlichen Zuschlag entsteht spätestens mit Beginn des zweiten Kalenderjahres, das der Annahme dieser Bestimmung durch Volk und Stände folgt.
3 Das Gesetz stellt sicher, dass der jährliche Zuschlag weder zu einer Reduktion der Ergänzungsleistungen noch zum Verlust des Anspruchs auf diese Leistungen führt.
1 SR 101
2 Die endgültige Ziffer dieser Übergangsbestimmung wird nach der Volksabstimmung von der Bundeskanzlei festgelegt.
```

Processed Text:

```plaintext
Art. [ARTICLE NUMBER HERE]
12. Übergangsbestimmung zu Art. 112 (Alters-, Hinterlassenen- und Invalidenversicherung)
1 Bezügerinnen und Bezüger einer Altersrente haben Anspruch auf einen jährlichen Zuschlag in der Höhe eines Zwölftels ihrer jährlichen Rente.
2 Der Anspruch auf den jährlichen Zuschlag entsteht spätestens mit Beginn des zweiten Kalenderjahres, das der Annahme dieser Bestimmung durch Volk und Stände folgt.
3 Das Gesetz stellt sicher, dass der jährliche Zuschlag weder zu einer Reduktion der Ergänzungsleistungen noch zum Verlust des Anspruchs auf diese Leistungen führt.
```

Extracted Article Numbers:

```plaintext
["197"]
```

[/Examples]

[Text to be processed]
"""

# %% Call openai model
output_messages = []
for i in range(len(data_train)):
    training_wortlaut = data_train[i]["Wortlaut"]

    text_to_beprocessed = prompt + training_wortlaut + "\n\n[/Text to be processed]"

    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        temperature=0.2,
        messages=[
            {"role": "system", "content": text_to_beprocessed},
        ],
        response_format=ModifiedLawText,
    )

    message = completion.choices[0].message

    if not message.parsed:
        raise ValueError("message was not parsed")

    output_messages.append(message.parsed)
    print(f"Completed {i+1}/{len(data_train)}")
    print(message.parsed)

# %% Save output
with open("output.pickle", "wb") as f:
    pickle.dump(output_messages, f)
# %% load output and print all the outputs like:

with open("output.pickle", "rb") as f:
    loaded_output = pickle.load(f)

assert len(loaded_output) == len(data_train)


for i in range(len(loaded_output)):
    print(f"train_{i+1} = [")
    print('r"""')
    print(data_train[i]["Wortlaut"])
    print('""", ')
    print('r"""')
    print(loaded_output[i].processed_text)
    print('""", ')
    print("[")
    print(
        ", ".join([f'"{x.number}"' for x in loaded_output[i].extracted_article_numbers])
    )
    print("]]")
# turn all the valid data into a list of dictionaries
# save the list of dictionaries into a pickle file


# %% Save valid data pickle
valid_list = [valid_1, valid_2, valid_3, valid_4, valid_5]

truncated_wortlaut_valid = []
for valid in valid_list:
    # destruct into wortlaut_original, wortlaut_truncated, article_numbers
    wortlaut_original, wortlaut_truncated, article_numbers = valid

    truncated_wortlaut_valid.append(
        {
            "wortlaut_original": wortlaut_original,
            "wortlaut_truncated": wortlaut_truncated,
            "article_numbers": article_numbers,
        }
    )

print(len(truncated_wortlaut_valid))

with open("../../data/processed/truncated_wortlaut_valid.pickle", "wb") as f:
    pickle.dump(truncated_wortlaut_valid, f)


# %% save the train data into a pickle file
truncated_wortlaut_valid = []
for i in range(len(loaded_output)):
    truncated_wortlaut_valid.append(
        {
            "wortlaut_original": data_train[i]["Wortlaut"],
            "wortlaut_truncated": loaded_output[i].processed_text,
            "article_numbers": loaded_output[i].extracted_article_numbers,
        }
    )


with open("../../data/processed/truncated_wortlaut_train.pickle", "wb") as f:
    pickle.dump(truncated_wortlaut_valid, f)


# %% sanity check all the saved data
with open("output.pickle", "rb") as f:
    loaded_output = pickle.load(f)

assert len(loaded_output) == len(data_train)

for i in range(len(loaded_output)):
    assert isinstance(loaded_output[i], ModifiedLawText)
    assert isinstance(loaded_output[i].processed_text, str)
    assert isinstance(loaded_output[i].extracted_article_numbers, list)
    assert all(
        isinstance(num, ArticleNumber)
        for num in loaded_output[i].extracted_article_numbers
    )
    assert loaded_output[i].processed_text != ""
    assert len(loaded_output[i].extracted_article_numbers) > 0

with open("../../data/processed/truncated_wortlaut_valid.pickle", "rb") as f:
    valid_data = pickle.load(f)

assert len(valid_data) == 5
for item in valid_data:
    assert "wortlaut_original" in item
    assert "wortlaut_truncated" in item
    assert "article_numbers" in item
    assert isinstance(item["wortlaut_original"], str)
    assert isinstance(item["wortlaut_truncated"], str)
    assert isinstance(item["article_numbers"], list)
    assert item["wortlaut_original"] != ""
    assert item["wortlaut_truncated"] != ""
    assert len(item["article_numbers"]) > 0

with open("../../data/processed/truncated_wortlaut_train.pickle", "rb") as f:
    train_data = pickle.load(f)

assert len(train_data) == len(data_train)
for item in train_data:
    assert "wortlaut_original" in item
    assert "wortlaut_truncated" in item
    assert "article_numbers" in item
    assert isinstance(item["wortlaut_original"], str)
    assert isinstance(item["wortlaut_truncated"], str)
    assert isinstance(item["article_numbers"], list)
    assert item["wortlaut_original"] != ""
    assert item["wortlaut_truncated"] != ""
    assert len(item["article_numbers"]) > 0

print("All sanity checks passed!")
