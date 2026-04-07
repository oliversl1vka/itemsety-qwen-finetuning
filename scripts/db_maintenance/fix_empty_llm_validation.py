#!/usr/bin/env python3
"""
Retroactive fix for validation bug: LLM producing 0 itemsets while Apriori found >0
was incorrectly marked as validation_passed = True.

This script:
1. Updates runs.db: validation_passed 1 → 0 for affected rows
2. Updates artifacts/validation_reports/*.json: injects llm_empty_output error
3. Updates artifacts/db_prepared/*.json: validation_passed true → false, llm_errors 0 → 1
4. Updates logs/validation/*.json: llm_errors 0 → 1, adds error sample

Usage:
    python scripts/db_maintenance/fix_empty_llm_validation.py          # Dry run
    python scripts/db_maintenance/fix_empty_llm_validation.py --apply  # Apply changes
"""
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "runs.db"

FIND_QUERY = """
SELECT id, dataset_id, llm_model, apriori_itemset_count, llm_itemset_count,
       validation_report_path
FROM runs
WHERE validation_passed = 1
  AND llm_itemset_count = 0
  AND apriori_itemset_count > 0
ORDER BY id
"""

LLM_EMPTY_ERROR = {
    "type": "llm_empty_output",
    "source": "llm",
    "detail": "LLM extracted 0 itemsets while Apriori found itemsets",
    "retroactive_fix": True,
    "fixed_at": None,  # filled at runtime
}


def derive_db_prepared_path(validation_report_path: str) -> str:
    """Derive db_prepared path from validation_report path."""
    return validation_report_path.replace(
        "validation_reports/", "db_prepared/"
    ).replace("_validation_report_", "_db_prepared_")


def derive_validation_log_path(validation_report_path: str) -> str:
    """Derive validation log path from validation_report path."""
    return validation_report_path.replace(
        "artifacts/validation_reports/", "logs/validation/"
    ).replace("_validation_report_", "_validation_generation_log_")


def fix_validation_report(filepath: Path, apriori_count: int, fix_ts: str, apply: bool) -> bool:
    """Inject llm_empty_output error into the validation report JSON."""
    if not filepath.exists():
        print(f"  [SKIP] Validation report not found: {filepath}")
        return False

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    llm_errors = data.get("llm", {}).get("errors", [])
    # Check if already fixed
    if any(e.get("type") == "llm_empty_output" for e in llm_errors):
        print(f"  [SKIP] Already fixed: {filepath.name}")
        return False

    error = {**LLM_EMPTY_ERROR, "apriori_count": apriori_count, "fixed_at": fix_ts}
    data["llm"]["errors"].append(error)

    if apply:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [{'FIXED' if apply else 'WOULD FIX'}] {filepath.name}")
    return True


def fix_db_prepared(filepath: Path, fix_ts: str, apply: bool) -> bool:
    """Update db_prepared JSON: validation_passed false, llm_errors 1."""
    if not filepath.exists():
        print(f"  [SKIP] db_prepared not found: {filepath}")
        return False

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("validation_passed") is False:
        print(f"  [SKIP] Already fixed: {filepath.name}")
        return False

    data["validation_passed"] = False
    data["llm_errors"] = 1
    data["retroactive_fix"] = True
    data["retroactive_fix_at"] = fix_ts

    if apply:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [{'FIXED' if apply else 'WOULD FIX'}] {filepath.name}")
    return True


def fix_validation_log(filepath: Path, apriori_count: int, fix_ts: str, apply: bool) -> bool:
    """Update validation log JSON: llm_errors 1, add error sample."""
    if not filepath.exists():
        print(f"  [SKIP] Validation log not found: {filepath}")
        return False

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("llm_errors", 0) > 0:
        print(f"  [SKIP] Already fixed: {filepath.name}")
        return False

    data["llm_errors"] = 1
    data["llm_error_sample"] = [{
        "type": "llm_empty_output",
        "apriori_count": apriori_count,
        "detail": "LLM extracted 0 itemsets while Apriori found itemsets",
        "retroactive_fix": True,
    }]
    data["retroactive_fix_at"] = fix_ts

    if apply:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [{'FIXED' if apply else 'WOULD FIX'}] {filepath.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fix empty-LLM validation bug retroactively")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry run)")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Path to runs.db")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    fix_ts = datetime.now(UTC).isoformat()
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"=== Retroactive fix for empty-LLM validation bug ({mode}) ===")
    print(f"Database: {db_path}")
    print(f"Timestamp: {fix_ts}\n")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows = cur.execute(FIND_QUERY).fetchall()
    print(f"Found {len(rows)} affected runs.\n")

    if not rows:
        print("Nothing to fix!")
        conn.close()
        return

    # Track stats
    stats = {
        "db_rows_fixed": 0,
        "validation_reports_fixed": 0,
        "db_prepared_fixed": 0,
        "validation_logs_fixed": 0,
        "skipped": 0,
    }

    for row in rows:
        run_id = row["id"]
        dataset_id = row["dataset_id"]
        apriori_count = row["apriori_itemset_count"]
        val_report_rel = row["validation_report_path"]

        print(f"--- Run {run_id}: {dataset_id} (apriori={apriori_count}, llm=0) ---")

        # 1. Validation report
        if val_report_rel:
            val_path = ROOT / val_report_rel
            if fix_validation_report(val_path, apriori_count, fix_ts, args.apply):
                stats["validation_reports_fixed"] += 1

            # 2. db_prepared
            db_prep_rel = derive_db_prepared_path(val_report_rel)
            db_prep_path = ROOT / db_prep_rel
            if fix_db_prepared(db_prep_path, fix_ts, args.apply):
                stats["db_prepared_fixed"] += 1

            # 3. Validation log
            val_log_rel = derive_validation_log_path(val_report_rel)
            val_log_path = ROOT / val_log_rel
            if fix_validation_log(val_log_path, apriori_count, fix_ts, args.apply):
                stats["validation_logs_fixed"] += 1
        else:
            print(f"  [SKIP] No validation_report_path in DB for run {run_id}")
            stats["skipped"] += 1

    # 4. Update database
    affected_ids = [row["id"] for row in rows]
    print(f"\n--- Database update: {len(affected_ids)} rows ---")
    if args.apply:
        placeholders = ",".join("?" * len(affected_ids))
        cur.execute(
            f"UPDATE runs SET validation_passed = 0 WHERE id IN ({placeholders})",
            affected_ids,
        )
        conn.commit()
        print(f"  [FIXED] Updated {cur.rowcount} rows: validation_passed = 0")
        stats["db_rows_fixed"] = cur.rowcount
    else:
        print(f"  [WOULD FIX] {len(affected_ids)} rows: validation_passed 1 → 0")
        print(f"  IDs: {affected_ids}")
        stats["db_rows_fixed"] = len(affected_ids)

    conn.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary ({mode}):")
    print(f"  DB rows fixed:              {stats['db_rows_fixed']}")
    print(f"  Validation reports fixed:   {stats['validation_reports_fixed']}")
    print(f"  db_prepared files fixed:    {stats['db_prepared_fixed']}")
    print(f"  Validation logs fixed:      {stats['validation_logs_fixed']}")
    print(f"  Skipped (already fixed/missing): {stats['skipped']}")
    print(f"{'='*60}")

    if not args.apply:
        print("\nThis was a DRY RUN. Add --apply to make changes.")


if __name__ == "__main__":
    main()
