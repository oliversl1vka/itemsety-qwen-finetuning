#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

files_to_fix = [
    "archive-12_TimeProvince.csv",
    "archive-12_TimeProvince_1.csv",
    "archive-22_data job posts.csv",
    "archive-22_data job posts_1.csv",
    "archive-23_online-job-postings.csv",
    "archive-23_online-job-postings_1.csv",
    "archive-36_segments.csv",
    "archive-9_groceries - groceries.csv",
    "newdata.csv",
    "store_data.csv"
]

print("🔧 FIXING INSUFFICIENT CATEGORICAL COLUMNS")
print("=" * 60)

for filename in files_to_fix:
    filepath = Path("real_datasets") / filename
    if not filepath.exists():
        print(f"❌ {filename}: Not found")
        continue
    
    try:
        df = pd.read_csv(filepath, low_memory=False)
        original_shape = df.shape
        
        cat_cols = []
        num_cols = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                if df[col].nunique() <= 100:
                    cat_cols.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                if df[col].nunique() <= 10:
                    cat_cols.append(col)
                elif df[col].nunique() > 10:
                    num_cols.append(col)
        
        needed = max(0, 3 - len(cat_cols))
        
        if needed == 0:
            print(f"✅ {filename}: Already has {len(cat_cols)} categorical cols")
            continue
        
        print(f"🔧 {filename}: Has {len(cat_cols)} cat cols, need {needed} more")
        
        binned = 0
        for col in num_cols[:needed + 3]:
            try:
                nuniq = df[col].nunique()
                if nuniq > 5:
                    try:
                        n_bins = min(5, max(3, nuniq // 50))
                        df[col] = pd.qcut(df[col], q=n_bins, 
                                         labels=[f'{col[:8]}_Q{i+1}' for i in range(n_bins)], 
                                         duplicates='drop')
                        cat_cols.append(col)
                        binned += 1
                    except:
                        n_bins = 4
                        df[col] = pd.cut(df[col], bins=n_bins, 
                                        labels=[f'{col[:8]}_B{i+1}' for i in range(n_bins)])
                        cat_cols.append(col)
                        binned += 1
                    
                    if len(cat_cols) >= 3:
                        break
            except:
                continue
        
        if len(cat_cols) >= 3:
            df.to_csv(filepath, index=False)
            print(f"   ✅ Fixed: {original_shape} → {df.shape}, now {len(cat_cols)} categorical ({binned} binned)")
        else:
            print(f"   ❌ Still only {len(cat_cols)} categorical columns")
            
    except Exception as e:
        print(f"❌ {filename}: Error - {str(e)[:60]}")

print("\n✅ Fixing complete - ready to generate datasets")
