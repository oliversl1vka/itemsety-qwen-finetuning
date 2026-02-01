"""
Vymazanie všetkých artefaktov a záznamov pre model gpt_4o_mini.
"""
import sqlite3
import json
from pathlib import Path

print("=" * 80)
print("MAZANIE ARTEFAKTOV A ZÁZNAMOV PRE gpt_4o_mini")
print("=" * 80)

# 1. Load database to get artifact paths
conn = sqlite3.connect('runs.db')
cursor = conn.cursor()

# Get all runs for gpt_4o_mini
cursor.execute('''
    SELECT id, dataset_name, apriori_output_path, llm_output_path, 
           validation_report_path
    FROM runs 
    WHERE llm_model = ?
''', ('gpt_4o_mini',))

runs = cursor.fetchall()
print(f"\nNájdených runov pre gpt_4o_mini: {len(runs)}")

if len(runs) == 0:
    print("Žiadne runy na zmazanie.")
    conn.close()
    exit(0)

# 2. Collect all artifact files
artifact_files = set()
for run in runs:
    run_id, dataset_name, apriori_path, llm_path, validation_path = run
    print(f"\nRun ID {run_id}: {dataset_name}")
    
    # Add all paths (some might be None)
    paths = [apriori_path, llm_path, validation_path]
    for path in paths:
        if path and Path(path).exists():
            artifact_files.add(Path(path))
            print(f"  → {path}")

# Also find all gpt_4o_mini artifacts in directories
artifact_dirs = [
    Path('artifacts/apriori_outputs'),
    Path('artifacts/extractor_outputs'),
    Path('artifacts/validation_reports'),
    Path('artifacts/db_prepared'),
    Path('logs/apriori'),
    Path('logs/extractor')
]

for artifact_dir in artifact_dirs:
    if artifact_dir.exists():
        for file in artifact_dir.glob('gpt_4o_mini_*'):
            artifact_files.add(file)

print(f"\n{'='*80}")
print(f"CELKOM SÚBOROV NA ZMAZANIE: {len(artifact_files)}")
print(f"{'='*80}")

# Show first 10 files
print("\nPríklady súborov (prvých 10):")
for i, file in enumerate(sorted(artifact_files)[:10]):
    print(f"  {i+1}. {file.name}")
if len(artifact_files) > 10:
    print(f"  ... a ďalších {len(artifact_files) - 10} súborov")

# 3. Delete artifact files
print(f"\n{'='*80}")
print("MAZANIE SÚBOROV...")
print(f"{'='*80}")

deleted_count = 0
for file in artifact_files:
    try:
        if file.exists():
            file.unlink()
            deleted_count += 1
    except Exception as e:
        print(f"❌ Chyba pri mazaní {file.name}: {e}")

print(f"✓ Zmazaných súborov: {deleted_count}/{len(artifact_files)}")

# 4. Delete database records
print(f"\n{'='*80}")
print("MAZANIE ZÁZNAMOV Z DATABÁZY...")
print(f"{'='*80}")

cursor.execute('DELETE FROM runs WHERE llm_model = ?', ('gpt_4o_mini',))
deleted_runs = cursor.rowcount
conn.commit()

print(f"✓ Zmazaných záznamov z DB: {deleted_runs}")

# 5. Verify
cursor.execute('SELECT COUNT(*) FROM runs WHERE llm_model = ?', ('gpt_4o_mini',))
remaining = cursor.fetchone()[0]

cursor.execute('SELECT llm_model, COUNT(*) FROM runs GROUP BY llm_model ORDER BY llm_model')
print(f"\n{'='*80}")
print("FINÁLNY STAV DATABÁZY:")
print(f"{'='*80}")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} runov")

cursor.execute('SELECT COUNT(*) FROM runs')
total = cursor.fetchone()[0]
print(f"\n  CELKOM: {total} runov")

conn.close()

print(f"\n{'='*80}")
if remaining == 0 and deleted_count > 0:
    print("✅ ÚSPEŠNE VYMAZANÉ!")
    print(f"   • {deleted_runs} záznamov z databázy")
    print(f"   • {deleted_count} artefaktov zo súborového systému")
else:
    print("⚠️ KONTROLA:")
    print(f"   • Zostávajúce záznamy v DB: {remaining}")
    print(f"   • Zmazané artefakty: {deleted_count}")
print(f"{'='*80}")
