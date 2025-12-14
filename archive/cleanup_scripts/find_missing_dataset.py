"""Zisti, ktory dataset chyba pre gpt_5_0."""
import sqlite3

conn = sqlite3.connect('runs.db')
cursor = conn.cursor()

# Ziskaj vsetky datasety pre gpt_5_0
cursor.execute('''
    SELECT dataset_name 
    FROM runs 
    WHERE llm_model = ? 
    ORDER BY dataset_name
''', ('gpt_5_0',))

gpt5_datasets = [row[0] for row in cursor.fetchall()]
print(f"GPT-5.0 has {len(gpt5_datasets)} datasets")

# Vytvor zoznam ocakavanych datasetov
expected_datasets = [f'ds_{i:04d}_' for i in range(1, 101)]

# Najdi chybajuce
missing = []
for i in range(1, 101):
    prefix = f'ds_{i:04d}_'
    found = any(ds.startswith(prefix) for ds in gpt5_datasets)
    if not found:
        missing.append(prefix.replace('_', '', -1))  # Remove trailing underscore for display

print(f"\nMissing datasets for gpt_5_0: {len(missing)}")
if missing:
    print("Missing:")
    for m in missing:
        print(f"  {m}")
        
        # Zisti, ci existuje pre gpt_4_1
        cursor.execute('''
            SELECT dataset_name 
            FROM runs 
            WHERE llm_model = ? AND dataset_name LIKE ?
        ''', ('gpt_4_1', f'{m}%'))
        gpt4_match = cursor.fetchone()
        if gpt4_match:
            print(f"    -> Exists for gpt_4_1: {gpt4_match[0]}")

conn.close()
