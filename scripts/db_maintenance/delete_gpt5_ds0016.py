#!/usr/bin/env python3
"""
Zmaze run pre gpt_5_0 a dataset ds_0016
"""

import sqlite3
from pathlib import Path

DB_PATH = 'runs.db'

def delete_gpt5_ds0016():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Najdi run
    cursor.execute('''
        SELECT id, dataset_name, apriori_output_path, llm_output_path, validation_report_path
        FROM runs
        WHERE llm_model = 'gpt_5_0' AND dataset_name LIKE '%ds_0016%'
    ''')
    
    rows = cursor.fetchall()
    
    if not rows:
        print("❌ Nenašiel sa žiadny run pre gpt_5_0 a ds_0016")
        conn.close()
        return
    
    print(f"\n{'='*60}")
    print(f"MAZANIE gpt_5_0 RUN PRE ds_0016")
    print(f"{'='*60}\n")
    
    for row in rows:
        run_id, dataset_name, apriori_path, llm_path, validation_path = row
        
        print(f"Run ID {run_id}: {dataset_name}")
        
        # Zber artefakty
        artifact_paths = []
        if apriori_path:
            artifact_paths.append(apriori_path)
        if llm_path:
            artifact_paths.append(llm_path)
        if validation_path:
            artifact_paths.append(validation_path)
        
        # Najdi a zmaz subory
        deleted_count = 0
        for rel_path in artifact_paths:
            file_path = Path(rel_path)
            if file_path.exists():
                print(f"  → Mažem: {file_path}")
                file_path.unlink()
                deleted_count += 1
            else:
                print(f"  ⚠ Neexistuje: {file_path}")
        
        print(f"  ✓ Zmazaných súborov: {deleted_count}")
        
        # Zmaz z DB
        cursor.execute('DELETE FROM runs WHERE id = ?', (run_id,))
        print(f"  ✓ Zmazaný záznam z DB")
    
    conn.commit()
    
    # Overenie
    cursor.execute('SELECT COUNT(*) FROM runs WHERE llm_model = "gpt_4_1"')
    gpt4_count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM runs WHERE llm_model = "gpt_5_0"')
    gpt5_count = cursor.fetchone()[0]
    
    print(f"\n{'='*60}")
    print(f"FINÁLNY STAV DATABÁZY:")
    print(f"{'='*60}")
    print(f"  gpt_4_1: {gpt4_count} runov")
    print(f"  gpt_5_0: {gpt5_count} runov")
    print(f"\n  CELKOM: {gpt4_count + gpt5_count} runov")
    print(f"{'='*60}")
    print(f"✅ ÚSPEŠNE VYMAZANÉ!")
    print(f"{'='*60}\n")
    
    conn.close()

if __name__ == '__main__':
    delete_gpt5_ds0016()
