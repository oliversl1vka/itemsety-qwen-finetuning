#!/usr/bin/env python3
"""
Check repository status before fine-tuning preparation.
Verifies database, artifacts, and file structure.
"""

import sqlite3
from pathlib import Path
import json


def check_repo_status():
    """Comprehensive repository status check"""

    print("=" * 60)
    print("🔍 REPOSITORY STATUS CHECK")
    print("=" * 60)

    checks = {}

    # Check database
    print("\n📊 DATABASE:")
    try:
        conn = sqlite3.connect("runs.db")
        cursor = conn.cursor()

        # Total runs
        cursor.execute("SELECT COUNT(*) FROM runs")
        total_runs = cursor.fetchone()[0]
        checks["database_exists"] = True
        checks["total_runs"] = total_runs
        print(f"   ✅ Database found: {total_runs} total runs")

        # Runs by model
        cursor.execute("SELECT llm_model, COUNT(*) FROM runs GROUP BY llm_model")
        for model, count in cursor.fetchall():
            print(f"      - {model}: {count} runs")

        # Validated runs
        cursor.execute("SELECT COUNT(*) FROM runs WHERE validation_passed = 1")
        valid_runs = cursor.fetchone()[0]
        checks["valid_runs"] = valid_runs
        print(f"   ✅ Validated runs: {valid_runs}")

        conn.close()
    except Exception as e:
        checks["database_exists"] = False
        print(f"   ❌ Database error: {e}")

    # Check datasets
    print("\n📁 DATASETS:")
    datasets_dir = Path("datasets")
    if datasets_dir.exists():
        csv_files = list(datasets_dir.glob("ds_*.csv"))
        checks["dataset_count"] = len(csv_files)
        print(f"   ✅ Found {len(csv_files)} dataset files")
    else:
        checks["dataset_count"] = 0
        print(f"   ❌ datasets/ directory not found")

    # Check artifacts
    print("\n🗂️  ARTIFACTS:")
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        for subdir in ["apriori_outputs", "extractor_outputs", "validation_reports"]:
            path = artifacts_dir / subdir
            if path.exists():
                count = len(list(path.glob("*.json")))
                print(f"   ✅ {subdir}: {count} files")
                checks[f"{subdir}_count"] = count
            else:
                print(f"   ⚠️  {subdir}: directory not found")
                checks[f"{subdir}_count"] = 0
    else:
        print(f"   ❌ artifacts/ directory not found")

    # Check system prompt
    print("\n📝 SYSTEM PROMPT:")
    prompt_file = Path("extractor_system_prompt.md")
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = len(f.readlines())
        checks["system_prompt_exists"] = True
        print(f"   ✅ extractor_system_prompt.md: {lines} lines")
    else:
        checks["system_prompt_exists"] = False
        print(f"   ❌ extractor_system_prompt.md not found")

    # Check requirements
    print("\n📦 DEPENDENCIES:")
    req_file = Path("requirements.txt")
    if req_file.exists():
        with open(req_file, "r") as f:
            reqs = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
        checks["requirements_exists"] = True
        print(f"   ✅ requirements.txt: {len(reqs)} packages")

        # Check for HF packages
        hf_packages = ["datasets", "transformers", "trl", "peft"]
        missing = [pkg for pkg in hf_packages if not any(pkg in req for req in reqs)]
        if missing:
            print(f"   ⚠️  Missing HF packages: {', '.join(missing)}")
            checks["hf_packages_ready"] = False
        else:
            print(f"   ✅ All HF training packages present")
            checks["hf_packages_ready"] = True
    else:
        checks["requirements_exists"] = False
        print(f"   ❌ requirements.txt not found")

    # Overall status
    print("\n" + "=" * 60)
    print("📋 OVERALL STATUS:")

    ready = (
        checks.get("database_exists", False)
        and checks.get("valid_runs", 0) >= 100
        and checks.get("dataset_count", 0) >= 100
        and checks.get("system_prompt_exists", False)
    )

    if ready:
        print("   ✅ Repository READY for fine-tuning preparation!")
    else:
        print("   ⚠️  Repository needs attention before fine-tuning")
        if checks.get("valid_runs", 0) < 100:
            print("      - Need at least 100 validated runs")
        if checks.get("dataset_count", 0) < 100:
            print("      - Need at least 100 datasets")
        if not checks.get("system_prompt_exists", False):
            print("      - Missing system prompt file")

    print("=" * 60)

    return checks


if __name__ == "__main__":
    check_repo_status()
