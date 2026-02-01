import sqlite3
conn = sqlite3.connect("runs.db")
cur = conn.cursor()
cur.execute("PRAGMA foreign_keys=ON;")
cur.execute("DELETE FROM runs WHERE id = ?", (101,))
print("Rows deleted:", cur.rowcount)
conn.commit()
conn.close()