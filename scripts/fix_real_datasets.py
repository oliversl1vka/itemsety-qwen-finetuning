#!/usr/bin/env python3
"""
Fix real_datasets directory:
1. Move purely numeric/problematic files to archive
2. Aggressively fix salvageable files with transformations
"""

import shutil
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    real_datasets_dir = Path("real_datasets")
    archive_dir = Path("archive/unsuitable_real_datasets")
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    suitable = []
    fixed = []
    moved = []
    
    print("🔍 AUDITING AND FIXING REAL DATASETS")
    print("=" * 60)
    
    for csv_file in sorted(real_datasets_dir.glob("*.csv")):
        size_mb = csv_file.stat().st_size / (1024 * 1024)
        
        # Move files that are too large (>100MB)
        if size_mb > 100:
            print(f"🗑️  {csv_file.name}: Too large ({size_mb:.1f}MB) - archiving")
            shutil.move(str(csv_file), str(archive_dir / csv_file.name))
            moved.append(csv_file.name)
            continue
        
        # Try to load and assess
        df = None
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                nrows = 10000 if size_mb > 50 else None
                df = pd.read_csv(csv_file, nrows=nrows, low_memory=False, encoding=encoding)
                break
            except:
                continue
        
        if df is None:
            print(f"❌ {csv_file.name}: Cannot load - archiving")
            shutil.move(str(csv_file), str(archive_dir / csv_file.name))
            moved.append(csv_file.name)
            continue
        
        # Count categorical columns
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
        
        # If already suitable, keep as-is
        if len(cat_cols) >= 3:
            suitable.append(csv_file.name)
            continue
        
        # Try to fix by binning numeric columns
        if len(cat_cols) < 3 and len(num_cols) >= (3 - len(cat_cols)):
            print(f"🔧 {csv_file.name}: Binning {3-len(cat_cols)} numeric columns...")
            
            bins_added = 0
            for col in num_cols[:(3-len(cat_cols))+2]:  # Try a few extra
                try:
                    if df[col].nunique() > 5:
                        df[col] = pd.qcut(df[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                        cat_cols.append(col)
                        bins_added += 1
                        if len(cat_cols) >= 3:
                            break
                except:
                    try:
                        df[col] = pd.cut(df[col], bins=4, labels=['B1', 'B2', 'B3', 'B4'])
                        cat_cols.append(col)
                        bins_added += 1
                        if len(cat_cols) >= 3:
                            break
                    except:
                        pass
            
            if len(cat_cols) >= 3:
                df.to_csv(csv_file, index=False)
                fixed.append(csv_file.name)
                print(f"   ✅ Fixed with {bins_added} binned columns")
            else:
                print(f"   ❌ Still only {len(cat_cols)} categorical - archiving")
                shutil.move(str(csv_file), str(archive_dir / csv_file.name))
                moved.append(csv_file.name)
        else:
            # Purely numeric or too few columns - archive
            print(f"🗑️  {csv_file.name}: Purely numeric - archiving")
            shutil.move(str(csv_file), str(archive_dir / csv_file.name))
            moved.append(csv_file.name)
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"  ✅ Already suitable: {len(suitable)}")
    print(f"  🔧 Fixed: {len(fixed)}")
    print(f"  🗑️  Archived: {len(moved)}")
    print(f"  📁 Total remaining: {len(suitable) + len(fixed)}")

if __name__ == "__main__":
    main()
