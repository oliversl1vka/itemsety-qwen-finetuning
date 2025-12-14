"""Zmaz riadky 201 a 222 z databazy."""
import sqlite3

conn = sqlite3.connect('runs.db')
cursor = conn.cursor()

# Najprv skontroluj, ci existuju
cursor.execute('SELECT id, dataset_name, llm_model FROM runs WHERE id IN (201, 222)')
rows = cursor.fetchall()
print(f"Found {len(rows)} rows to delete:")
for row in rows:
    print(f"  ID {row[0]}: {row[1]} ({row[2]})")

# Zmaz ich
cursor.execute('DELETE FROM runs WHERE id IN (201, 222)')
deleted = cursor.rowcount
print(f"\nDeleted {deleted} rows")

conn.commit()
conn.close()

print("Done!")
