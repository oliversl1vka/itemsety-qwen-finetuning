"""
Analyze all datasets in real_datasets/, convert ARFF to CSV, 
select top 25 most suitable for frequent itemset mining, and remove others.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import shutil
from scipy.io import arff
from collections import Counter

# Setup
real_datasets_dir = Path("real_datasets")
results = []

def analyze_dataset(file_path: Path) -> dict:
    """Analyze a single dataset and return suitability metrics."""
    try:
        # Load dataset
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path, low_memory=False, nrows=1000)  # Sample for speed
        else:
            return None
        
        total_cols = len(df.columns)
        total_rows = len(df)
        
        # Count column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Analyze categorical vs continuous
        categorical_cols = []
        continuous_cols = []
        
        for col in numeric_cols:
            unique_count = df[col].nunique()
            if unique_count <= 10:  # Likely categorical
                categorical_cols.append(col)
            else:
                continuous_cols.append(col)
        
        # Object columns - check if truly categorical
        for col in object_cols:
            unique_count = df[col].nunique()
            if unique_count <= 50:  # Reasonable for itemset mining
                categorical_cols.append(col)
        
        # Calculate suitability score
        categorical_ratio = len(categorical_cols) / total_cols if total_cols > 0 else 0
        
        # Penalty for too many unique values
        avg_unique = np.mean([df[col].nunique() for col in categorical_cols]) if categorical_cols else 0
        
        # Bonus for boolean/binary columns
        binary_cols = [col for col in categorical_cols if df[col].nunique() == 2]
        binary_ratio = len(binary_cols) / total_cols if total_cols > 0 else 0
        
        # Calculate final score (0-100)
        score = (
            categorical_ratio * 50 +  # 50% weight on categorical ratio
            binary_ratio * 30 +        # 30% bonus for binary columns
            (1 - min(avg_unique / 50, 1)) * 20  # 20% penalty for many unique values
        )
        
        # Determine suitability tier
        if score >= 70:
            tier = "A_EXCELLENT"
        elif score >= 50:
            tier = "B_GOOD"
        elif score >= 30:
            tier = "C_FAIR"
        else:
            tier = "D_POOR"
        
        return {
            "filename": file_path.name,
            "rows": total_rows,
            "total_cols": total_cols,
            "categorical_cols": len(categorical_cols),
            "continuous_cols": len(continuous_cols),
            "binary_cols": len(binary_cols),
            "categorical_ratio": round(categorical_ratio, 3),
            "binary_ratio": round(binary_ratio, 3),
            "score": round(score, 2),
            "tier": tier,
            "categorical_names": categorical_cols[:10],  # Sample
        }
        
    except Exception as e:
        return {
            "filename": file_path.name,
            "error": str(e),
            "score": 0,
            "tier": "ERROR"
        }


# Convert ARFF files first
print("="*80)
print("STEP 1: Converting ARFF files to CSV")
print("="*80)

arff_files = list(real_datasets_dir.glob("*.arff"))
for arff_file in arff_files:
    try:
        print(f"\nConverting: {arff_file.name}")
        data, meta = arff.loadarff(arff_file)
        df = pd.DataFrame(data)
        
        # Decode byte strings to regular strings
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].str.decode('utf-8')
                except:
                    pass
        
        csv_path = arff_file.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Created: {csv_path.name} ({len(df)} rows, {len(df.columns)} cols)")
        
        # Delete ARFF file
        arff_file.unlink()
        print(f"🗑️  Deleted: {arff_file.name}")
        
    except Exception as e:
        print(f"❌ Error converting {arff_file.name}: {e}")

# Analyze all CSV files
print("\n" + "="*80)
print("STEP 2: Analyzing all datasets")
print("="*80)

csv_files = sorted(real_datasets_dir.glob("*.csv"))
print(f"\nFound {len(csv_files)} CSV files\n")

for csv_file in csv_files:
    print(f"Analyzing: {csv_file.name}...", end=" ")
    result = analyze_dataset(csv_file)
    if result:
        results.append(result)
        print(f"{result['tier']} (score: {result['score']})")
    else:
        print("SKIPPED")

# Sort by score
results.sort(key=lambda x: x['score'], reverse=True)

# Print results
print("\n" + "="*80)
print("STEP 3: Ranking Results")
print("="*80)
print(f"\n{'Rank':<5} {'Score':<7} {'Tier':<12} {'Cat%':<7} {'Bin%':<7} {'Cols':<6} {'Rows':<8} {'Filename':<50}")
print("-"*110)

for i, result in enumerate(results, 1):
    if 'error' not in result:
        print(f"{i:<5} {result['score']:<7.2f} {result['tier']:<12} "
              f"{result['categorical_ratio']:<7.3f} {result['binary_ratio']:<7.3f} "
              f"{result['total_cols']:<6} {result['rows']:<8} {result['filename']:<50}")

# Select top 25
top_25 = results[:25]
to_delete = results[25:]

print("\n" + "="*80)
print(f"STEP 4: Selecting Top 25 (deleting {len(to_delete)} files)")
print("="*80)

print("\n✅ KEEPING (Top 25):")
for i, dataset in enumerate(top_25, 1):
    if 'error' not in dataset:
        print(f"  {i:2}. {dataset['filename']:<50} (score: {dataset['score']:.1f}, tier: {dataset['tier']})")

print(f"\n❌ DELETING ({len(to_delete)}):")
for dataset in to_delete:
    file_path = real_datasets_dir / dataset['filename']
    if file_path.exists():
        print(f"  🗑️  {dataset['filename']:<50} (score: {dataset.get('score', 0):.1f})")
        file_path.unlink()

# Save analysis report
report = {
    "total_analyzed": len(results),
    "top_25_selected": [d['filename'] for d in top_25 if 'error' not in d],
    "deleted": [d['filename'] for d in to_delete],
    "tier_distribution": dict(Counter([d['tier'] for d in top_25 if 'error' not in d])),
    "detailed_results": top_25
}

with open("dataset_analysis_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✅ Total analyzed: {len(results)}")
print(f"✅ Top 25 kept: {len(top_25)}")
print(f"❌ Deleted: {len(to_delete)}")
print(f"📊 Tier distribution: {dict(Counter([d['tier'] for d in top_25 if 'error' not in d]))}")
print(f"💾 Report saved: dataset_analysis_report.json")
print("\n✅ DONE! Ready for training data generation.")
