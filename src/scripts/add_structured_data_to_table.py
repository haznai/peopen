# scenario:
# we ran `create_db_for_volksabstimmungen.py` and `structured_data_from_broshures.py` successfully.
# we now have `volksabstimmungen.db` and `volksabstimmung_structured` folder.
# let's combine the parses structured data with the database.
# the goal is to have a final database with metadate info about the volksabstimmung (date, yes/no votes)
# with the parsed arguments from the brochures (`volksabstimmung_structured`)

import json
import os
import sqlite3
from evaluate import load

# load all structured data as json into memory, with date (dir name) as key, and title of the volksabstimmung as second key
structured_data: dict[str, dict[str, dict]] = {}
base_path = "data/volksabstimmung_structured"
for date_folder in os.listdir(base_path):
    # turn 2020_09_27 into 2020-09-27 for db comparison
    comparable_date = date_folder.replace("_", "-")
    structured_data[comparable_date] = {}
    for file in os.listdir(base_path + "/" + date_folder):
        with open(base_path + "/" + date_folder + "/" + file) as f:
            parsed_json = json.load(f)['parsed']
            structured_data[comparable_date][parsed_json["titel"]] = parsed_json

# load the database into memory
conn = sqlite3.connect('data/processed/volksabstimmungen.db')

# fix some date formatting issues in the database
# the following "Datum" entries in the database have a wrong format (M and D are swapped)
# swap the month and date back to the correct format
wrong_dates_in_database = [
    "2020-09-02",
    "2019-10-02",
    "2021-07-03",
    "2024-09-06",
]

for wrong_date in wrong_dates_in_database:
    # swap the month and date back to the correct format
    year, month, day = wrong_date.split("-")
    correct_date = f"{year}-{day}-{month}"
    conn.execute(f"UPDATE volksabstimmungen SET Datum = '{correct_date}' WHERE Datum = '{wrong_date}'")

# there's  a "Datum" column and a "Titel der Vorlage" column in the database
# create a dict with the date as key and the titles list as value
database_data: dict[str, list[str]] = {}
cursor = conn.execute("SELECT Datum, `Titel der Vorlage` FROM volksabstimmungen")
for row in cursor:
    if row[0] not in database_data:
        database_data[row[0]] = []
    database_data[row[0]].append(row[1])


# do a berscore comparison of the titles in the database and the structured data
# only compare the entries that are in the same broshure (date)
bertscore = load("bertscore")

# matches between structured data and database
# key is the date, value is a second dict with the structured data title as key and the database tile as value
matches: dict[str, dict[str, str]] = {}
for date in structured_data:
    if date in database_data:
        structured_data_entries = structured_data[date]
        database_entries = database_data[date]
        matches[date] = {}
        # compare all structured data entries with all database entries
        # and assign the best match to the database entry
        # this is a naive approach, but it's good enough for this use case
        for structured_data_title in structured_data_entries:
            best_match = None
            best_score = 0
            for database_title in database_entries:
                score = bertscore.compute(
                    predictions=[structured_data_title],
                    references=[database_title],
                    model_type="bert-base-multilingual-cased",
                    lang=["de"],
                )
                score = sum(score["f1"]) / len(score["f1"])
                if score > best_score:
                    best_score = score
                    best_match = database_title
            if best_match:
                matches[date][structured_data_title] = best_match
    else:
        raise ValueError(f"No database entries for {date}")

# pretty print all the matches
# for date in matches:
#     print(date)
#     for structured_data_title, database_title in matches[date].items():
#         print(f"   * {structured_data_title} -> {database_title}")


# there's a single match that is wrong, change it manually
matches["2020-02-09"]["Verbot der Diskriminierung aufgrund der sexuellen Orientierung"] = "Änderung vom 14. Dezember 2018 des Strafgesetzbuches und des Militärstrafgesetzes"
# assert that there's just two entries for the date still
assert len(matches["2020-02-09"]) == 2

# now, add the structured data to the database
# we add the structured data as json to the database
# the json is stored in the "Daten aus der Broschüre" column
# add the column if it doesn't exist
conn.execute("ALTER TABLE volksabstimmungen ADD COLUMN `Daten aus der Broschüre` TEXT")
for date in matches:
    for structured_data_title, database_title in matches[date].items():
        structured_data_json = structured_data[date][structured_data_title]
        structured_data_json = json.dumps(structured_data_json)
        structured_data_json = structured_data_json.replace("'", "''")
        database_title = database_title.replace("'", "''")
        conn.execute(f"UPDATE volksabstimmungen SET `Daten aus der Broschüre` = ? WHERE `Titel der Vorlage` = ? and Datum = ?", (structured_data_json, database_title, date))
        # ensure that the data was added correctly
        # this is a bit overkill, but it's good to be sure
        cursor = conn.execute("SELECT `Daten aus der Broschüre` FROM volksabstimmungen WHERE `Titel der Vorlage` = ? and Datum = ?", (database_title, date))
        for row in cursor:
            assert row[0] == structured_data_json

# save the changes to the database
conn.commit()
conn.close()
