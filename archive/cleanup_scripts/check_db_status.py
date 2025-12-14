"""Zobraz stav databazy."""
import sqlite3

conn = sqlite3.connect('runs.db')
cursor = conn.cursor()

cursor.execute('SELECT llm_model, COUNT(*) FROM runs GROUP BY llm_model ORDER BY llm_model')
print('Models in database:')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]} rows')

cursor.execute('SELECT COUNT(*) FROM runs')
print(f'\nTotal rows: {cursor.fetchone()[0]}')

conn.close()
