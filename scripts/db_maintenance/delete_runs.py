import sqlite3

conn = sqlite3.connect('runs.db')
cursor = conn.cursor()

# Najprv si pozrieme, koľko runov máme
cursor.execute('SELECT COUNT(*) FROM runs')
total_before = cursor.fetchone()[0]
print(f'Total runs before deletion: {total_before}')

# Získame ID runov 101-200 (zoradené podľa timestamp)
cursor.execute('''
    SELECT id FROM (
        SELECT id, ROW_NUMBER() OVER (ORDER BY timestamp ASC) as row_num
        FROM runs
    )
    WHERE row_num BETWEEN 101 AND 200
''')
ids_to_delete = [row[0] for row in cursor.fetchall()]
print(f'Found {len(ids_to_delete)} runs to delete (IDs 101-200)')

if ids_to_delete:
    # Zmažeme runy
    placeholders = ','.join(['?'] * len(ids_to_delete))
    cursor.execute(f'DELETE FROM runs WHERE id IN ({placeholders})', ids_to_delete)
    conn.commit()
    print(f'Deleted {cursor.rowcount} runs')
    
    # Overíme výsledok
    cursor.execute('SELECT COUNT(*) FROM runs')
    total_after = cursor.fetchone()[0]
    print(f'Total runs after deletion: {total_after}')
else:
    print('No runs to delete')

conn.close()
