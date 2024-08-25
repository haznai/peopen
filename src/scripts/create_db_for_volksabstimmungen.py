import pandas as pd
import sqlite3

# load csv into pd dataframe
# "Datum" is the first column and a date
df = pd.read_csv("../../data/processed/table_of_each_volksabstimmung_since_new_broshure.csv")

# convert "Datum" to datetime, then into iso format string
df["Datum"] = pd.to_datetime(df["Datum"], format="mixed").dt.strftime("%Y-%m-%d")


# save data in database for later use
# Connect to SQLite database
conn = sqlite3.connect('volksabstimmungen.db')

# Write DataFrame to SQLite database
df.to_sql('volksabstimmungen', conn, if_exists='replace', index=False)

# Close the connection
conn.close()
