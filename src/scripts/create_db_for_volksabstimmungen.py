import pandas as pd
import sqlite3
import shutil

# Load CSV into pd dataframe
df = pd.read_csv("../../data/processed/volksabstimmungen_hand_filled_iniatives.csv")

# Create a copy of the original database
original_db = "../../data/processed/volksabstimmungen.db"
new_db = "../../data/processed/volksiniativen_with_wortlaut.db"
shutil.copy2(original_db, new_db)

# Connect to the new SQLite database
conn = sqlite3.connect(new_db)

# Filter df to entries where "Wortlaut" is not null
df = df[df["Wortlaut"].notna()]

# Read existing data from the database
db_df = pd.read_sql_query(
    'SELECT rowid, Datum, art, "Titel der Vorlage" FROM volksabstimmungen',
    conn,
)

# Merge the database DataFrame with the CSV DataFrame
merged_df = pd.merge(
    db_df,
    df[["Datum", "art", "Titel der Vorlage", "Wortlaut"]],
    on=["Datum", "art", "Titel der Vorlage"],
    how="inner",
)

# Sanity checks
print(f"Original CSV rows: {len(df)}")
print(f"Database rows: {len(db_df)}")
print(f"Merged rows: {len(merged_df)}")

if len(merged_df) == 0:
    print("Warning: No matching rows found!")
elif len(merged_df) < min(len(df), len(db_df)):
    print("Warning: Some rows did not match. Check your data.")

# Add Wortlaut column if it doesn't exist
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(volksabstimmungen)")
columns = [column[1] for column in cursor.fetchall()]
if "Wortlaut" not in columns:
    cursor.execute("ALTER TABLE volksabstimmungen ADD COLUMN Wortlaut TEXT")

# Update the database with the new Wortlaut values
cursor.executemany(
    "UPDATE volksabstimmungen SET Wortlaut = ? WHERE rowid = ?",
    merged_df[["Wortlaut", "rowid"]].values,
)

print(f"Updated {cursor.rowcount} rows in the database.")

# Remove entries that weren't matched/merged
unmatched_rowids = set(db_df["rowid"]) - set(merged_df["rowid"])
if unmatched_rowids:
    placeholders = ",".join("?" for _ in unmatched_rowids)
    cursor.execute(
        f"DELETE FROM volksabstimmungen WHERE rowid IN ({placeholders})",
        list(unmatched_rowids),
    )
    print(f"Removed {cursor.rowcount} unmatched rows from the database.")

# Commit changes and close the connection
conn.commit()
conn.close()

print(f"Updates have been saved to the new database: {new_db}")
