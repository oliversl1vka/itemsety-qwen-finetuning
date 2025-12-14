"""Zmaz vsetky riadky s modelom gpt_4o_mini z databazy."""
import sqlite3

conn = sqlite3.connect('runs.db')
cursor = conn.cursor()

# Najprv skontroluj, kolko ich je
cursor.execute('SELECT COUNT(*) FROM runs WHERE llm_model = ?', ('gpt_4o_mini',))
count = cursor.fetchone()[0]
print(f"Found {count} rows with model gpt_4o_mini")

if count > 0:
    # Zobraz ich
    cursor.execute('SELECT id, dataset_name, llm_model FROM runs WHERE llm_model = ? ORDER BY id', ('gpt_4o_mini',))
    rows = cursor.fetchall()
    print("\nRows to delete:")
    for row in rows[:10]:  # Show first 10
        print(f"  ID {row[0]}: {row[1]} ({row[2]})")
    if len(rows) > 10:
        print(f"  ... and {len(rows) - 10} more")
    
    # Zmaz ich
    cursor.execute('DELETE FROM runs WHERE llm_model = ?', ('gpt_4o_mini',))
    deleted = cursor.rowcount
    print(f"\nDeleted {deleted} rows with model gpt_4o_mini")
    
    conn.commit()
    print("Changes committed!")
else:
    print("No rows to delete.")

conn.close()
print("Done!")
