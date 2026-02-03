#!/usr/bin/env python3
"""
Push training datasets to HuggingFace Space for RLHF training.

Dataset structure:
- CSV files (original transaction data)
- Apriori outputs (ground truth itemsets)
- LLM extractions (model predictions)
- Human feedback (LLM-generated annotations for RLHF)
"""

import argparse
import json
import shutil
import os
from pathlib import Path
from datetime import datetime, UTC
from huggingface_hub import HfApi, CommitOperationAdd
import sqlite3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SPACE_ID = "OliverSlivka/testrun2"
HF_TOKEN = os.getenv("HF_TOKEN")

def load_runs_from_db(db_path: Path, llm_model: str = None, min_itemsets: int = 1) -> list:
    """Load validated runs from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = """
        SELECT dataset_id, data_path, apriori_itemset_count, llm_itemset_count,
               apriori_output_path, llm_output_path, validation_report_path,
               timestamp, llm_model
        FROM runs
        WHERE validation_passed = 1
          AND llm_itemset_count >= ?
    """
    
    params = [min_itemsets]
    
    if llm_model:
        query += " AND llm_model = ?"
        params.append(llm_model)
    
    query += " ORDER BY timestamp DESC"
    
    cursor.execute(query, params)
    
    runs = []
    for row in cursor.fetchall():
        runs.append({
            "dataset_id": row[0],
            "dataset_path": row[1],
            "apriori_count": row[2],
            "llm_count": row[3],
            "apriori_output_path": row[4],
            "llm_output_path": row[5],
            "validation_report_path": row[6],
            "timestamp": row[7],
            "llm_model": row[8],
        })
    
    conn.close()
    return runs


def generate_human_feedback_annotation(apriori_data: dict, llm_data: dict, validation_report: dict) -> dict:
    """
    Generate LLM-based human feedback annotation for RLHF.
    
    Compares Apriori (ground truth) with LLM output and creates feedback signals.
    """
    apriori_itemsets = set(tuple(sorted(item["itemset"])) for item in apriori_data)
    llm_itemsets = set(tuple(sorted(item["itemset"])) for item in llm_data)
    
    # Calculate metrics
    true_positives = apriori_itemsets & llm_itemsets
    false_positives = llm_itemsets - apriori_itemsets
    false_negatives = apriori_itemsets - llm_itemsets
    
    precision = len(true_positives) / len(llm_itemsets) if llm_itemsets else 0
    recall = len(true_positives) / len(apriori_itemsets) if apriori_itemsets else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Generate feedback
    feedback = {
        "quality_score": f1,  # 0-1 score for RLHF reward
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "correct_itemsets": [list(itemset) for itemset in true_positives],
        "incorrect_itemsets": [list(itemset) for itemset in false_positives],
        "missed_itemsets": [list(itemset) for itemset in false_negatives],
        "validation_passed": validation_report.get("all_passed", False),
        "feedback_type": "automated_comparison",
        "reward_signal": f1,  # Use F1 as reward for RLHF
        "annotations": {
            "correct": len(true_positives),
            "incorrect": len(false_positives),
            "missed": len(false_negatives),
            "total_ground_truth": len(apriori_itemsets),
            "total_predicted": len(llm_itemsets),
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }
    
    return feedback


def prepare_rlhf_dataset(runs: list, artifacts_dir: Path) -> list:
    """
    Prepare dataset in RLHF format: csv + apriori + llm + human_feedback.
    """
    rlhf_examples = []
    
    for run in runs:
        dataset_path = Path(run["dataset_path"])
        if not dataset_path.exists():
            print(f"  ⚠️ Dataset not found: {dataset_path}")
            continue
        
        # Load artifacts - paths already include 'artifacts/' prefix
        apriori_path = Path(run["apriori_output_path"])
        llm_path = Path(run["llm_output_path"])
        validation_path = Path(run["validation_report_path"])
        
        if not all([apriori_path.exists(), llm_path.exists(), validation_path.exists()]):
            # Debug missing files
            continue
        
        with open(apriori_path) as f:
            apriori_data = json.load(f)
        with open(llm_path) as f:
            llm_data = json.load(f)
        with open(validation_path) as f:
            validation_report = json.load(f)
        
        # Generate human feedback annotation
        human_feedback = generate_human_feedback_annotation(
            apriori_data, llm_data, validation_report
        )
        
        # Create RLHF example
        rlhf_example = {
            "dataset_id": run["dataset_id"],
            "csv_path": str(dataset_path),
            "apriori": apriori_data,
            "llm_output": llm_data,
            "human_feedback": human_feedback,
            "metadata": {
                "timestamp": run["timestamp"],
                "llm_model": run["llm_model"],
                "apriori_count": run["apriori_count"],
                "llm_count": run["llm_count"],
            }
        }
        
        rlhf_examples.append(rlhf_example)
    
    return rlhf_examples


def test_dataset_quality(rlhf_examples: list, min_f1: float = 0.3) -> dict:
    """
    Test dataset quality before pushing to HF Space.
    
    Returns test results with pass/fail status.
    """
    results = {
        "total_examples": len(rlhf_examples),
        "passed": 0,
        "failed": 0,
        "avg_f1": 0.0,
        "min_f1_required": min_f1,
        "test_passed": False,
        "issues": [],
    }
    
    if len(rlhf_examples) == 0:
        results["issues"].append("No examples to test")
        return results
    
    f1_scores = []
    for example in rlhf_examples:
        f1 = example["human_feedback"]["f1_score"]
        f1_scores.append(f1)
        
        if f1 >= min_f1:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["issues"].append(f"Dataset {example['dataset_id']} has low F1: {f1:.2f}")
    
    results["avg_f1"] = sum(f1_scores) / len(f1_scores)
    
    # Pass if at least 80% of examples meet minimum F1
    pass_rate = results["passed"] / results["total_examples"]
    results["pass_rate"] = pass_rate
    results["test_passed"] = pass_rate >= 0.8
    
    if not results["test_passed"]:
        results["issues"].append(f"Pass rate {pass_rate:.1%} is below 80%")
    
    return results


def push_to_space(rlhf_examples: list, artifacts_dir: Path, test_before_push: bool = True) -> bool:
    """
    Push dataset to HuggingFace Space.
    
    Structure pushed to Space:
    - datasets/ (CSV files)
    - rlhf_data.jsonl (RLHF training data)
    - metadata.json (dataset statistics)
    """
    
    # Test dataset quality
    if test_before_push:
        print("\n🧪 Testing dataset quality...")
        test_results = test_dataset_quality(rlhf_examples)
        
        print(f"   Total examples: {test_results['total_examples']}")
        print(f"   Passed: {test_results['passed']} ({test_results['pass_rate']:.1%})")
        print(f"   Failed: {test_results['failed']}")
        print(f"   Average F1: {test_results['avg_f1']:.3f}")
        
        if not test_results["test_passed"]:
            print("\n❌ Dataset quality test FAILED:")
            for issue in test_results["issues"]:
                print(f"   - {issue}")
            return False
        
        print("✅ Dataset quality test PASSED")
    
    # Prepare files for upload
    print("\n📦 Preparing files for upload...")
    
    api = HfApi(token=HF_TOKEN)
    operations = []
    
    # 1. Create RLHF data file
    rlhf_jsonl = "\n".join([json.dumps(ex) for ex in rlhf_examples])
    operations.append(
        CommitOperationAdd(
            path_in_repo="rlhf_data.jsonl",
            path_or_fileobj=rlhf_jsonl.encode()
        )
    )
    
    # 2. Copy CSV files to datasets/ folder in Space
    csv_files_copied = 0
    for example in rlhf_examples:
        csv_path = Path(example["csv_path"])
        if csv_path.exists():
            with open(csv_path, "rb") as f:
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=f"datasets/{csv_path.name}",
                        path_or_fileobj=f.read()
                    )
                )
                csv_files_copied += 1
    
    # 3. Create metadata file
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_examples": len(rlhf_examples),
        "csv_files": csv_files_copied,
        "avg_f1_score": test_dataset_quality(rlhf_examples, min_f1=0.0)["avg_f1"],
        "structure": {
            "csv": "Original transaction data",
            "apriori": "Ground truth frequent itemsets",
            "llm_output": "LLM-extracted itemsets",
            "human_feedback": "Automated feedback for RLHF (precision, recall, F1)"
        },
        "usage": "Use rlhf_data.jsonl for RLHF training. Each example contains reward signals.",
    }
    
    operations.append(
        CommitOperationAdd(
            path_in_repo="metadata.json",
            path_or_fileobj=json.dumps(metadata, indent=2).encode()
        )
    )
    
    # 4. Push to Space
    print(f"\n📤 Pushing to HuggingFace Space: {SPACE_ID}")
    print(f"   Files to upload: {len(operations)}")
    
    try:
        api.create_commit(
            repo_id=SPACE_ID,
            repo_type="space",
            operations=operations,
            commit_message=f"Add {len(rlhf_examples)} RLHF training examples ({datetime.now(UTC).strftime('%Y-%m-%d %H:%M')})"
        )
        
        print(f"✅ Successfully pushed to Space!")
        print(f"   View at: https://huggingface.co/spaces/{SPACE_ID}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to push to Space: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Push RLHF training datasets to HuggingFace Space")
    parser.add_argument("--db", type=Path, default="runs.db",
                       help="Path to runs database")
    parser.add_argument("--artifacts-dir", type=Path, default="artifacts",
                       help="Path to artifacts directory")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini",
                       help="Filter by LLM model")
    parser.add_argument("--min-itemsets", type=int, default=1,
                       help="Minimum number of itemsets required")
    parser.add_argument("--min-f1", type=float, default=0.3,
                       help="Minimum F1 score for quality test")
    parser.add_argument("--skip-test", action="store_true",
                       help="Skip quality testing before push")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PUSH TO HUGGINGFACE SPACE - RLHF DATASET PREPARATION")
    print("=" * 70)
    
    # Load validated runs
    print(f"\n📂 Loading validated runs from {args.db}...")
    runs = load_runs_from_db(args.db, args.model, args.min_itemsets)
    print(f"   Found {len(runs)} validated runs")
    
    if len(runs) == 0:
        print("❌ No runs found. Run pipeline first.")
        return 1
    
    # Prepare RLHF dataset
    print("\n🔨 Preparing RLHF dataset (csv + apriori + llm + human_feedback)...")
    rlhf_examples = prepare_rlhf_dataset(runs, args.artifacts_dir)
    print(f"   Prepared {len(rlhf_examples)} RLHF examples")
    
    if len(rlhf_examples) == 0:
        print("❌ No valid examples prepared. Check artifact paths.")
        return 1
    
    # Push to Space
    success = push_to_space(
        rlhf_examples,
        args.artifacts_dir,
        test_before_push=not args.skip_test
    )
    
    print("\n" + "=" * 70)
    if success:
        print("✅ PUSH COMPLETE!")
        print(f"   Space URL: https://huggingface.co/spaces/{SPACE_ID}")
        print(f"   Examples: {len(rlhf_examples)}")
        print("\n   Files in Space:")
        print("   - rlhf_data.jsonl (training data)")
        print("   - datasets/*.csv (original CSVs)")
        print("   - metadata.json (statistics)")
    else:
        print("❌ PUSH FAILED - see errors above")
        return 1
    
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
