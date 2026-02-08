#!/usr/bin/env python3
"""Fast dataset generator - preprocesses source data once"""

import hashlib
import json
import random
from pathlib import Path
from datetime import datetime, UTC
import pandas as pd
import numpy as np

def main():
    random.seed(42)
    np.random.seed(42)
    
    source_dir = Path("real_datasets")
    output_dir = Path("data/datasets_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    existing = len(list(output_dir.glob("ds_*.csv")))
    target = 500
    
    print(f"🔄 FAST DATASET GENERATOR")
    print(f"📁 Existing: {existing}, Target: {target}")
    
    # Load sources ONCE
    sources = []
    for csv_file in sorted(source_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            
            # Count categorical columns (including low-cardinality numeric)
            cat_cols = []
            num_cols = []
            
            for c in df.columns:
                if df[c].dtype == 'object' and df[c].nunique() <= 100:
                    cat_cols.append(c)
                elif df[c].dtype in ['int64', 'float64']:
                    if df[c].nunique() <= 10:
                        cat_cols.append(c)
                    elif df[c].nunique() > 10:
                        num_cols.append(c)
            
            # If not enough categorical, try binning numeric
            if len(cat_cols) < 3 and len(num_cols) >= (3 - len(cat_cols)):
                for col in num_cols[:(3 - len(cat_cols)) + 2]:
                    try:
                        n_bins = 4
                        df[col] = pd.cut(df[col], bins=n_bins, labels=[f'B{i+1}' for i in range(n_bins)])
                        cat_cols.append(col)
                        if len(cat_cols) >= 3:
                            break
                    except:
                        pass
            
            if len(cat_cols) >= 3:
                sources.append((csv_file.name, df, cat_cols[:20]))  # Limit to 20 cols
                print(f"✅ {csv_file.name}: {len(cat_cols)} categorical columns")
        except Exception as e:
            print(f"❌ {csv_file.name}: {str(e)[:50]}")
            pass
    
    if not sources:
        print("❌ No suitable sources found!")
        return
    
    print(f"\n📦 Loaded {len(sources)} sources")
    print(f"🎯 Generating {target - existing} new datasets...\n")
    
    # Generate
    ds_idx = existing + 1
    variations = ['subsample', 'shuffle', 'noise']
    
    while ds_idx <= target:
        for src_name, src_df, cat_cols in sources:
            if ds_idx > target:
                break
            
            # Pick variation
            var = random.choice(variations)
            
            # Sample rows (5-25)
            n_rows = random.randint(5, min(25, len(src_df)))
            sampled = src_df.sample(n=n_rows, replace=False).copy()
            
            # Sample columns (5-15)
            n_cols = random.randint(5, min(15, len(cat_cols)))
            sel_cols = random.sample(cat_cols, n_cols)
            sampled = sampled[sel_cols]
            
            # Add noise if selected
            if var == 'noise' and len(sampled) > 2:
                for col in sel_cols[:2]:  # Add noise to 2 columns
                    idx = random.randint(0, len(sampled)-1)
                    sampled.iloc[idx, sampled.columns.get_loc(col)] = 'noise_value'
            
            # Save
            csv_str = sampled.to_csv(index=False)
            hash_val = hashlib.sha256(csv_str.encode()).hexdigest()[:8]
            filename = f"ds_{ds_idx:04d}_{len(sampled)}x{len(sel_cols)}_{hash_val}.csv"
            
            (output_dir / filename).write_text(csv_str)
            
            if ds_idx % 25 == 0:
                print(f"  ✓ {ds_idx}/{target} datasets generated...")
            
            ds_idx += 1
    
    print(f"\n✅ COMPLETE: {ds_idx-1} total datasets in {output_dir}")

if __name__ == "__main__":
    main()
