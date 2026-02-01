"""Check final database state for presentation."""
import sqlite3

conn = sqlite3.connect('runs.db')
cursor = conn.cursor()

print("=" * 60)
print("FINAL DATABASE STATE FOR PRESENTATION")
print("=" * 60)

# Models and counts
cursor.execute('SELECT llm_model, COUNT(*) FROM runs GROUP BY llm_model ORDER BY llm_model')
print("\nModels in database:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} runs")

# Validation stats
cursor.execute('SELECT COUNT(*) FROM runs WHERE validation_passed = 1')
passed = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM runs')
total = cursor.fetchone()[0]
print(f"\nValidation: {passed}/{total} passed ({100*passed/total:.1f}%)")

# Itemset statistics
cursor.execute('''
    SELECT 
        AVG(llm_itemset_count), 
        MIN(llm_itemset_count), 
        MAX(llm_itemset_count),
        AVG(apriori_itemset_count)
    FROM runs 
    WHERE validation_passed = 1
''')
stats = cursor.fetchone()
print(f"\nItemset statistics (LLM):")
print(f"  Average: {stats[0]:.2f}")
print(f"  Min: {stats[1]}")
print(f"  Max: {stats[2]}")
print(f"  Apriori avg: {stats[3]:.2f}")

# Dataset dimensions
cursor.execute('''
    SELECT dataset_name FROM runs LIMIT 1
''')
print(f"\nTotal runs: {total}")

conn.close()
