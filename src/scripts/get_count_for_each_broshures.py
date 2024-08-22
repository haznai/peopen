# Counts the number of things to vote on in each broshure

import pandas as pd

# load csv into pd dataframe
# "Datum" is the first column and a date
df = pd.read_csv("../../data/processed/table_of_each_volksabstimmung_since_new_broshure.csv")


# convert "Datum" to datetime, then into iso format string
df["Datum"] = pd.to_datetime(df["Datum"], format="mixed").dt.strftime("%Y-%m-%d")


# count the number of volksinitiatives in each broshure
# by "Datum" column, sorted
df.groupby("Datum").count()["art"]
