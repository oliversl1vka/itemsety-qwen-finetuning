#!/usr/bin/env python3
"""Final cleanup: Archive remaining unsuitable files"""

import shutil
import pandas as pd
from pathlib import Path

real_datasets_dir = Path("real_datasets")
archive_dir = Path("archive/unsuitable_real_datasets")
archive_dir.mkdir(parents=True, exist_ok=True)

suitable = []
to_archive = []

print("🔍 FINAL AUDIT\n" + "=" * 60)

for csv_file in sorted(real_datasets_dir.glob("*.csv")):
    try:
        size_mb = csv_file.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            to_archive.append(csv_file)
            continue
        
        df = pd.read_csv(csv_file, nrows=100, low_memory=False)
        cat_cols = sum(1 for c in df.columns 
                      if (df[c].dtype == 'object' and df[c].nunique() <= 100)
                      or (df[c].dtype in ['int64', 'float64'] and df[c].nunique() <= 10))
        
        if cat_cols >= 3:
            suitable.append(csv_file.name)
        else:
            to_archive.append(csv_file)
    except:
        to_archive.append(csv_file)

print(f"✅ Suitable files: {len(suitable)}")
print(f"🗑️  To archive: {len(to_archive)}")

if to_archive:
    print("\nArchiving unsuitable files...")
    for f in to_archive:
        shutil.move(str(f), str(archive_dir / f.name))
        print(f"  → {f.name}")

print(f"\n📊 FINAL COUNT: {len(suitable)} suitable datasets ready for generation")
