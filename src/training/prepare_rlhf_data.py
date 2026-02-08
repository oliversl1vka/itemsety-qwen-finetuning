#!/usr/bin/env python3
"""
Prepare RLHF-style datasets from Apriori + LLM extractions.

Based on best practices from awesome-RLHF repository:
https://github.com/opendilab/awesome-RLHF

Supported formats:
1. Pairwise Preference (chosen vs rejected)
2. Ranking (multiple responses ranked by quality)
3. Dense Reward (F1-based reward signals)
"""

import argparse
import json
import sqlite3
from pathlib import Path
from datetime import datetime, UTC
from typing import Dict, List, Tuple
import hashlib


def load_runs_from_db(db_path: Path, min_itemsets: int = 1, llm_model: str = None) -> List[Dict]:
    """Load validated runs from SQLite database."""
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


def compute_metrics(apriori_itemsets: set, llm_itemsets: set) -> Dict:
    """Compute precision, recall, F1 between Apriori and LLM outputs."""
    true_positives = apriori_itemsets & llm_itemsets
    false_positives = llm_itemsets - apriori_itemsets
    false_negatives = apriori_itemsets - llm_itemsets
    
    precision = len(true_positives) / len(llm_itemsets) if llm_itemsets else 0
    recall = len(true_positives) / len(apriori_itemsets) if apriori_itemsets else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": len(true_positives),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
    }


def create_pairwise_preference(run: Dict, apriori_data: List, llm_data: List, 
                                metrics: Dict) -> Dict:
    """
    Create pairwise preference format (like HH-RLHF, Anthropic datasets).
    
    Format:
    {
        "prompt": "Extract frequent itemsets from: <CSV data>",
        "chosen": <Apriori response - ground truth>,
        "rejected": <LLM response if F1 < threshold>,
        "chosen_reward": 1.0,
        "rejected_reward": <F1 score>
    }
    """
    # Read CSV data
    csv_path = Path(run["dataset_path"])
    with open(csv_path, "r") as f:
        csv_content = f.read()
    
    # Create prompt
    prompt = f"""Extract all frequent itemsets from the following CSV dataset.
Return them as a JSON array with itemsets, counts, and supporting row IDs.

Dataset:
{csv_content[:1000]}... (truncated)

Instructions:
- Find all frequently occurring item combinations
- Include itemset, count, support, and row evidence
- Return as structured JSON array"""
    
    # Chosen response (Apriori - ground truth)
    chosen_response = json.dumps(apriori_data, indent=2)
    
    # Rejected response (LLM output if worse than Apriori)
    rejected_response = json.dumps(llm_data, indent=2)
    
    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
        "chosen_reward": 1.0,  # Apriori is ground truth
        "rejected_reward": metrics["f1_score"],
        "metadata": {
            "dataset_id": run["dataset_id"],
            "metrics": metrics,
            "timestamp": run["timestamp"],
        }
    }


def create_ranking_format(run: Dict, apriori_data: List, llm_data: List,
                          metrics: Dict) -> Dict:
    """
    Create ranking format with multiple responses ranked by quality.
    
    Format:
    {
        "prompt": "...",
        "responses": [
            {"text": <response>, "rank": 1, "score": <reward>},
            {"text": <response>, "rank": 2, "score": <reward>},
        ]
    }
    """
    csv_path = Path(run["dataset_path"])
    with open(csv_path, "r") as f:
        csv_content = f.read()
    
    prompt = f"""Extract frequent itemsets from this CSV dataset:

{csv_content[:1000]}..."""
    
    return {
        "prompt": prompt,
        "responses": [
            {
                "text": json.dumps(apriori_data, indent=2),
                "rank": 1,
                "score": 1.0,
                "quality": "ground_truth",
                "method": "Apriori",
            },
            {
                "text": json.dumps(llm_data, indent=2),
                "rank": 2,
                "score": metrics["f1_score"],
                "quality": f"f1={metrics['f1_score']:.2f}",
                "method": "LLM",
            }
        ],
        "metadata": {
            "dataset_id": run["dataset_id"],
            "metrics": metrics,
        }
    }


def create_dense_reward_format(run: Dict, apriori_data: List, llm_data: List,
                                metrics: Dict) -> Dict:
    """
    Create dense reward format with itemset-level rewards.
    
    Format:
    {
        "prompt": "...",
        "response": <LLM response>,
        "rewards": [
            {"itemset": [...], "reward": 1.0, "correct": true},
            {"itemset": [...], "reward": 0.0, "correct": false},
        ],
        "global_reward": <F1 score>
    }
    """
    csv_path = Path(run["dataset_path"])
    with open(csv_path, "r") as f:
        csv_content = f.read()
    
    prompt = f"Extract frequent itemsets: {csv_content[:500]}..."
    
    # Create itemset-level rewards
    apriori_set = {tuple(sorted(item["itemset"])) for item in apriori_data}
    
    rewards = []
    for llm_item in llm_data:
        itemset = tuple(sorted(llm_item["itemset"]))
        is_correct = itemset in apriori_set
        
        rewards.append({
            "itemset": list(itemset),
            "reward": 1.0 if is_correct else 0.0,
            "correct": is_correct,
            "support": llm_item.get("support", 0),
        })
    
    return {
        "prompt": prompt,
        "response": json.dumps(llm_data, indent=2),
        "dense_rewards": rewards,
        "global_reward": metrics["f1_score"],
        "metrics": metrics,
        "metadata": {
            "dataset_id": run["dataset_id"],
            "total_itemsets": len(llm_data),
            "correct_itemsets": metrics["true_positives"],
        }
    }


def create_multi_objective_format(run: Dict, apriori_data: List, llm_data: List,
                                   metrics: Dict) -> Dict:
    """
    Multi-objective reward format (precision, recall, F1, completeness).
    
    Format:
    {
        "prompt": "...",
        "response": <LLM response>,
        "rewards": {
            "precision": <0-1>,
            "recall": <0-1>,
            "f1_score": <0-1>,
            "completeness": <0-1>,  # How many ground truth itemsets found
            "efficiency": <0-1>,    # Avoid false positives
        }
    }
    """
    csv_path = Path(run["dataset_path"])
    with open(csv_path, "r") as f:
        csv_content = f.read()
    
    prompt = f"Extract frequent itemsets: {csv_content[:500]}..."
    
    # Additional metrics
    completeness = metrics["recall"]  # Same as recall
    efficiency = 1.0 - (metrics["false_positives"] / len(llm_data) if llm_data else 0)
    
    return {
        "prompt": prompt,
        "response": json.dumps(llm_data, indent=2),
        "rewards": {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "completeness": completeness,
            "efficiency": efficiency,
        },
        "aggregate_reward": metrics["f1_score"],  # Primary reward
        "metadata": {
            "dataset_id": run["dataset_id"],
            "metrics": metrics,
        }
    }


def prepare_rlhf_datasets(runs: List[Dict], format_type: str = "pairwise") -> List[Dict]:
    """
    Prepare RLHF datasets in specified format.
    
    Args:
        runs: List of validated runs from database
        format_type: "pairwise", "ranking", "dense_reward", or "multi_objective"
    
    Returns:
        List of RLHF examples in chosen format
    """
    rlhf_examples = []
    
    for run in runs:
        # Load artifacts
        apriori_path = Path(run["apriori_output_path"])
        llm_path = Path(run["llm_output_path"])
        
        if not all([apriori_path.exists(), llm_path.exists()]):
            print(f"  ⚠️ Missing artifacts for {run['dataset_id']}")
            continue
        
        with open(apriori_path) as f:
            apriori_data = json.load(f)
        with open(llm_path) as f:
            llm_data = json.load(f)
        
        # Compute metrics
        apriori_set = {tuple(sorted(item["itemset"])) for item in apriori_data}
        llm_set = {tuple(sorted(item["itemset"])) for item in llm_data}
        metrics = compute_metrics(apriori_set, llm_set)
        
        # Create example in chosen format
        if format_type == "pairwise":
            example = create_pairwise_preference(run, apriori_data, llm_data, metrics)
        elif format_type == "ranking":
            example = create_ranking_format(run, apriori_data, llm_data, metrics)
        elif format_type == "dense_reward":
            example = create_dense_reward_format(run, apriori_data, llm_data, metrics)
        elif format_type == "multi_objective":
            example = create_multi_objective_format(run, apriori_data, llm_data, metrics)
        else:
            raise ValueError(f"Unknown format: {format_type}")
        
        rlhf_examples.append(example)
    
    return rlhf_examples


def save_datasets(examples: List[Dict], output_dir: Path, format_type: str):
    """Save RLHF datasets to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL
    output_file = output_dir / f"rlhf_{format_type}.jsonl"
    with open(output_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"✅ Saved {len(examples)} examples to {output_file}")
    
    # Save metadata
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "format_type": format_type,
        "total_examples": len(examples),
        "avg_f1": sum(ex.get("metadata", {}).get("metrics", {}).get("f1_score", 0) for ex in examples) / len(examples) if examples else 0,
        "description": {
            "pairwise": "Chosen (Apriori) vs Rejected (LLM) responses",
            "ranking": "Multiple responses ranked by quality",
            "dense_reward": "Itemset-level rewards",
            "multi_objective": "Multiple reward signals (precision, recall, F1)",
        }.get(format_type, "Unknown format"),
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dumps(metadata, indent=2, file=f)
    
    print(f"✅ Saved metadata to {output_dir / 'metadata.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare RLHF datasets from Apriori + LLM extractions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create pairwise preference dataset (like Anthropic HH-RLHF)
  python prepare_rlhf_data.py --format pairwise
  
  # Create ranking dataset
  python prepare_rlhf_data.py --format ranking
  
  # Create dense reward dataset (itemset-level rewards)
  python prepare_rlhf_data.py --format dense_reward
  
  # Create multi-objective reward dataset
  python prepare_rlhf_data.py --format multi_objective
  
  # Create all formats
  python prepare_rlhf_data.py --format all
        """
    )
    
    parser.add_argument("--db", type=Path, default="runs.db",
                       help="Path to runs database (default: runs.db)")
    parser.add_argument("--output-dir", type=Path, default="data/rlhf_datasets",
                       help="Output directory for RLHF datasets")
    parser.add_argument("--format", type=str, default="pairwise",
                       choices=["pairwise", "ranking", "dense_reward", "multi_objective", "all"],
                       help="RLHF dataset format")
    parser.add_argument("--min-itemsets", type=int, default=1,
                       help="Minimum itemsets required")
    parser.add_argument("--llm-model", type=str, default=None,
                       help="Filter by LLM model")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RLHF DATASET PREPARATION")
    print("=" * 80)
    print(f"Format: {args.format}")
    print(f"Database: {args.db}")
    print(f"Output: {args.output_dir}")
    
    # Load runs
    print(f"\n📂 Loading validated runs from database...")
    runs = load_runs_from_db(args.db, args.min_itemsets, args.llm_model)
    print(f"   Found {len(runs)} validated runs")
    
    if len(runs) == 0:
        print("❌ No validated runs found. Run pipeline first.")
        return 1
    
    # Prepare datasets
    formats = ["pairwise", "ranking", "dense_reward", "multi_objective"] if args.format == "all" else [args.format]
    
    for format_type in formats:
        print(f"\n🔨 Preparing {format_type} format...")
        examples = prepare_rlhf_datasets(runs, format_type)
        
        if len(examples) == 0:
            print(f"   ⚠️ No examples created for {format_type}")
            continue
        
        # Save datasets
        save_datasets(examples, args.output_dir / format_type, format_type)
    
    print("\n" + "=" * 80)
    print("✅ RLHF DATASET PREPARATION COMPLETE!")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Total runs processed: {len(runs)}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
