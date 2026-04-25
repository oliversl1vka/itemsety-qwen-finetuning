# 🧹 Repository Cleanup & HuggingFace Training Preparation Plan

**Date:** December 14, 2025  
**Goal:** Prepare repository for HuggingFace fine-tuning with Claude assistance

---

## 📊 CURRENT STATE ASSESSMENT

### Database Status ✅
- **Total Runs:** 300 (100 × gpt_4_1, 100 × gpt_4o_mini, 100 × gpt_5_0)
- **Validation:** 100% pass rate across all models
- **Schema:** 27 columns including `llm_model`, paths, metadata
- **Status:** READY FOR TRAINING DATA EXTRACTION

### File Structure ✅
```
✅ Core pipeline files (pipeline.py, dataset_generation.py, visualization.py)
✅ System prompt (extractor_system_prompt.md)
✅ Database (runs.db with 300 validated runs)
✅ Datasets (100 CSV files: ds_0001 to ds_0100)
✅ Artifacts (apriori_outputs, extractor_outputs, validation_reports)
✅ Documentation (README.md, PRESENTATION_GUIDE_FINAL.md)
⚠️ Cleanup scripts (8 files) - archive these
⚠️ Legacy presentation files (2 files) - archive
⚠️ Test file (ds_TEST.csv) - remove or relocate
```

### Files to Clean/Archive/Update
```
ARCHIVE (move to archive/):
- check_db_status.py
- check_final_state.py
- delete_gpt4omini.py
- delete_gpt4omini_all.py
- delete_gpt5_ds0016.py
- delete_rows.py
- delete_runs.py
- find_missing_dataset.py
- db_editor.py (keep as utility but document)
- PRESENTATION_CHEAT_SHEET.md (superseded by V2)
- create_presentation_visuals_v2.py (keep final only)

REMOVE:
- ds_TEST.csv (test file)
- __pycache__/ (add to .gitignore if not present)

UPDATE:
- requirements.txt (add HuggingFace dependencies)
- README.md (add fine-tuning section)
- .gitignore (ensure comprehensive)
- azure.env (add template version)
```

---

## 🎯 PHASE 1: FILE CLEANUP (30 minutes)

### Step 1.1: Create Archive Directory
```powershell
New-Item -ItemType Directory -Path "archive" -Force
New-Item -ItemType Directory -Path "archive/cleanup_scripts" -Force
New-Item -ItemType Directory -Path "archive/presentation_legacy" -Force
```

### Step 1.2: Move Cleanup Scripts
```powershell
Move-Item -Path "check_db_status.py" -Destination "archive/cleanup_scripts/"
Move-Item -Path "check_final_state.py" -Destination "archive/cleanup_scripts/"
Move-Item -Path "delete_gpt4omini.py" -Destination "archive/cleanup_scripts/"
Move-Item -Path "delete_gpt4omini_all.py" -Destination "archive/cleanup_scripts/"
Move-Item -Path "delete_gpt5_ds0016.py" -Destination "archive/cleanup_scripts/"
Move-Item -Path "delete_rows.py" -Destination "archive/cleanup_scripts/"
Move-Item -Path "delete_runs.py" -Destination "archive/cleanup_scripts/"
Move-Item -Path "find_missing_dataset.py" -Destination "archive/cleanup_scripts/"
```

### Step 1.3: Move Legacy Presentation Files
```powershell
Move-Item -Path "PRESENTATION_CHEAT_SHEET.md" -Destination "archive/presentation_legacy/"
Move-Item -Path "create_presentation_visuals_v2.py" -Destination "archive/presentation_legacy/"
```

### Step 1.4: Remove Test Files
```powershell
Remove-Item -Path "ds_TEST.csv" -Force
```

### Step 1.5: Clean Python Cache
```powershell
Remove-Item -Path "__pycache__" -Recurse -Force -ErrorAction SilentlyContinue
```

---

## 🔧 PHASE 2: UPDATE CORE FILES (1 hour)

### Step 2.1: Update `.gitignore`
**File:** `.gitignore`

**Add these entries:**
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
ENV/
env/

# Secrets (CRITICAL!)
azure.env
.env
*.key
*.pem

# Training artifacts (optional - depends if you want to track)
# training_dataset/
# checkpoints/
# runs/
# wandb/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Temporary/Test files
ds_TEST.csv
temp_*.csv
scratch_*.py

# Large files
*.db-journal
*.db-wal

# Archive
archive/
```

### Step 2.2: Create Azure Environment Template
**File:** `azure.env.template`

```bash
# Azure OpenAI Configuration Template
# Copy this file to azure.env and fill in your credentials
# NEVER commit azure.env to git!

# Azure OpenAI Endpoint (replace with your resource)
AZURE_OPENAI_ENDPOINT=https://func-aicoe-ap-oairest-prd-001.azurewebsites.net

# API Key (fill in your actual key)
AZURE_OPENAI_API_KEY=your-api-key-here

# API Version
AZURE_OPENAI_API_VERSION=2024-10-21

# Deployment Names
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-gpt-5_deployment1
AZURE_OPENAI_DEPLOYMENT=gpt-gpt-5_deployment1
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002_deployment0

# Optional: Other services
JIRA_URL=https://jira.rbinternational.com/jira/
JIRA_TOKEN=
CHROMA_TELEMETRY=false
```

### Step 2.3: Update `requirements.txt`
**File:** `requirements.txt`

**Current:**
```txt
langchain>=0.2.0
langchain-openai>=0.1.0
pandas>=2.0.0
python-dotenv>=1.0.0
faiss-cpu>=1.7.4
matplotlib>=3.8.0
```

**Add HuggingFace Dependencies:**
```txt
# Core pipeline dependencies
langchain>=0.2.0
langchain-openai>=0.1.0
pandas>=2.0.0
python-dotenv>=1.0.0
faiss-cpu>=1.7.4
matplotlib>=3.8.0

# Fine-tuning dependencies (for HuggingFace training)
datasets>=2.14.0
transformers>=4.35.0
trl>=0.7.0
peft>=0.7.0
accelerate>=0.24.0
bitsandbytes>=0.41.0

# Optional: Training monitoring
# wandb>=0.15.0
# tensorboard>=2.14.0
```

---

## 📝 PHASE 3: CREATE TRAINING PREPARATION SCRIPTS (2 hours)

### Step 3.1: Database Export Utility
**File:** `export_training_data.py`

```python
#!/usr/bin/env python3
"""
Export validated runs from runs.db for HuggingFace training.
Creates training-ready JSON files with CSV context + ground truth.
"""

import sqlite3
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import argparse

def load_csv_sample(csv_path: str, max_rows: int = 50) -> str:
    """Load CSV and format as text (limited rows to avoid token overflow)"""
    try:
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        sample_rows = min(max_rows, total_rows)
        
        # Format as readable transaction list
        rows_text = []
        for idx, row in df.head(sample_rows).iterrows():
            items = [str(v) for v in row.values if pd.notna(v) and str(v).strip()]
            rows_text.append(f"Row {idx+1}: {', '.join(items)}")
        
        header = f"Dataset: {Path(csv_path).name}\n"
        header += f"Dimensions: {total_rows} rows × {len(df.columns)} columns\n\n"
        
        if total_rows > sample_rows:
            footer = f"\n... ({total_rows - sample_rows} more rows not shown)"
        else:
            footer = ""
        
        return header + "\n".join(rows_text) + footer
    
    except Exception as e:
        return f"Error loading CSV: {e}"

def load_ground_truth(apriori_path: str) -> List[Dict[str, Any]]:
    """Load Apriori output (ground truth itemsets)"""
    try:
        with open(apriori_path, 'r', encoding='utf-8') as f:
            itemsets = json.load(f)
        
        # Clean format: keep only itemset, count, rows
        cleaned = []
        for item in itemsets:
            cleaned.append({
                "itemset": item["itemset"],
                "count": item["count"],
                "rows": item["rows"]
            })
        
        return cleaned
    
    except Exception as e:
        print(f"Warning: Failed to load {apriori_path}: {e}")
        return []

def export_training_examples(
    db_path: str = "runs.db",
    output_dir: str = "training_data",
    max_csv_rows: int = 50,
    model_filter: str = "gpt_4_1"  # Use one model to avoid duplicates
) -> None:
    """Export all validated runs as training examples"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Query validated runs
    cursor.execute("""
        SELECT 
            dataset_id,
            dataset_name,
            data_path,
            apriori_output_path,
            llm_output_path,
            min_support,
            dataset_size_rows,
            dataset_hash
        FROM runs
        WHERE validation_passed = 1
        AND llm_model = ?
        ORDER BY dataset_id
    """, (model_filter,))
    
    examples = []
    skipped = 0
    
    for row in cursor.fetchall():
        dataset_id = row['dataset_id']
        
        # Load CSV context
        csv_context = load_csv_sample(row['data_path'], max_csv_rows)
        
        # Load ground truth (Apriori output)
        ground_truth = load_ground_truth(row['apriori_output_path'])
        
        if not ground_truth:
            print(f"⚠️  Skipping {dataset_id}: No valid ground truth")
            skipped += 1
            continue
        
        # Create training example
        example = {
            "id": dataset_id,
            "dataset_name": row['dataset_name'],
            "dataset_hash": row['dataset_hash'],
            "csv_context": csv_context,
            "min_support": row['min_support'],
            "ground_truth": ground_truth,
            "metadata": {
                "total_rows": row['dataset_size_rows'],
                "itemset_count": len(ground_truth),
                "data_path": row['data_path'],
                "apriori_path": row['apriori_output_path']
            }
        }
        
        examples.append(example)
        
        # Save individual file
        example_file = output_path / f"{dataset_id}_training.json"
        with open(example_file, 'w', encoding='utf-8') as f:
            json.dump(example, f, indent=2, ensure_ascii=False)
    
    conn.close()
    
    # Save combined file
    combined_file = output_path / "all_training_examples.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary = {
        "total_examples": len(examples),
        "skipped": skipped,
        "model_filter": model_filter,
        "max_csv_rows": max_csv_rows,
        "output_directory": str(output_path)
    }
    
    summary_file = output_path / "export_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Export Complete!")
    print(f"   Examples: {len(examples)}")
    print(f"   Skipped: {skipped}")
    print(f"   Output: {output_path}")
    print(f"\nFiles created:")
    print(f"   - {len(examples)} individual JSON files")
    print(f"   - all_training_examples.json (combined)")
    print(f"   - export_summary.json")

def main():
    parser = argparse.ArgumentParser(description="Export training data from runs.db")
    parser.add_argument("--db", default="runs.db", help="Path to SQLite database")
    parser.add_argument("--output", default="training_data", help="Output directory")
    parser.add_argument("--max-rows", type=int, default=50, help="Max CSV rows to include")
    parser.add_argument("--model", default="gpt_4_1", help="Model to filter by")
    
    args = parser.parse_args()
    
    export_training_examples(
        db_path=args.db,
        output_dir=args.output,
        max_csv_rows=args.max_rows,
        model_filter=args.model
    )

if __name__ == "__main__":
    main()
```

### Step 3.2: HuggingFace Dataset Creator
**File:** `create_hf_dataset.py`

```python
#!/usr/bin/env python3
"""
Convert exported training data to HuggingFace Dataset format.
Supports conversational format for SFT training.
"""

import json
from pathlib import Path
from datasets import Dataset, DatasetDict
import argparse
from typing import List, Dict, Any

def create_conversational_example(
    example: Dict[str, Any],
    system_prompt_path: str = "extractor_system_prompt.md"
) -> Dict[str, Any]:
    """Convert raw example to conversational format for SFT"""
    
    # Load system prompt
    with open(system_prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    # Format user message
    user_message = f"{example['csv_context']}\n\n"
    user_message += f"Find all frequent itemsets with minimum support count = {example['min_support']}."
    
    # Format assistant response (ground truth)
    assistant_response = json.dumps(example['ground_truth'], ensure_ascii=False)
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ],
        "metadata": {
            "dataset_id": example['id'],
            "dataset_name": example['dataset_name'],
            "itemset_count": len(example['ground_truth']),
            "min_support": example['min_support']
        }
    }

def create_hf_dataset(
    input_file: str = "training_data/all_training_examples.json",
    output_dir: str = "hf_dataset",
    train_split: float = 0.9,
    system_prompt_path: str = "extractor_system_prompt.md"
) -> None:
    """Create HuggingFace dataset from training examples"""
    
    # Load examples
    with open(input_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    print(f"📦 Loaded {len(examples)} training examples")
    
    # Convert to conversational format
    conversational_examples = []
    for example in examples:
        conv_example = create_conversational_example(example, system_prompt_path)
        conversational_examples.append(conv_example)
    
    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(conversational_examples)
    
    split_idx = int(len(conversational_examples) * train_split)
    train_examples = conversational_examples[:split_idx]
    val_examples = conversational_examples[split_idx:]
    
    # Create datasets
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)
    
    # Create dataset dict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    # Save to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    
    print(f"\n✅ HuggingFace Dataset Created!")
    print(f"   Train: {len(train_dataset)} examples")
    print(f"   Validation: {len(val_dataset)} examples")
    print(f"   Output: {output_path}")
    print(f"\nTo load:")
    print(f"   from datasets import load_from_disk")
    print(f"   dataset = load_from_disk('{output_path}')")

def main():
    parser = argparse.ArgumentParser(description="Create HuggingFace dataset")
    parser.add_argument("--input", default="training_data/all_training_examples.json")
    parser.add_argument("--output", default="hf_dataset")
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--system-prompt", default="extractor_system_prompt.md")
    
    args = parser.parse_args()
    
    create_hf_dataset(
        input_file=args.input,
        output_dir=args.output,
        train_split=args.train_split,
        system_prompt_path=args.system_prompt
    )

if __name__ == "__main__":
    main()
```

### Step 3.3: Dataset Inspector
**File:** `inspect_training_data.py`

```python
#!/usr/bin/env python3
"""
Inspect and validate training data before fine-tuning.
Checks format, statistics, and potential issues.
"""

import json
from pathlib import Path
from datasets import load_from_disk
import argparse

def inspect_hf_dataset(dataset_path: str = "hf_dataset") -> None:
    """Inspect HuggingFace dataset structure and statistics"""
    
    dataset = load_from_disk(dataset_path)
    
    print("=" * 60)
    print("📊 DATASET INSPECTION REPORT")
    print("=" * 60)
    
    # Basic stats
    print(f"\n🔢 SPLIT SIZES:")
    print(f"   Train: {len(dataset['train'])} examples")
    print(f"   Validation: {len(dataset['validation'])} examples")
    print(f"   Total: {len(dataset['train']) + len(dataset['validation'])} examples")
    
    # Sample first example
    print(f"\n📝 SAMPLE EXAMPLE (train[0]):")
    sample = dataset['train'][0]
    print(f"   Messages: {len(sample['messages'])} turns")
    for i, msg in enumerate(sample['messages']):
        role = msg['role']
        content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
        print(f"     {i+1}. {role}: {content_preview}")
    
    # Metadata stats
    print(f"\n📈 METADATA STATISTICS:")
    if 'metadata' in sample:
        itemset_counts = [ex['metadata']['itemset_count'] for ex in dataset['train']]
        print(f"   Itemset count range: {min(itemset_counts)} - {max(itemset_counts)}")
        print(f"   Average itemsets: {sum(itemset_counts) / len(itemset_counts):.1f}")
    
    # Message length stats
    print(f"\n📏 MESSAGE LENGTH STATISTICS:")
    user_lengths = [len(ex['messages'][1]['content']) for ex in dataset['train']]
    assistant_lengths = [len(ex['messages'][2]['content']) for ex in dataset['train']]
    
    print(f"   User message (CSV context):")
    print(f"     Min: {min(user_lengths)} chars")
    print(f"     Max: {max(user_lengths)} chars")
    print(f"     Avg: {sum(user_lengths) / len(user_lengths):.0f} chars")
    
    print(f"   Assistant message (ground truth):")
    print(f"     Min: {min(assistant_lengths)} chars")
    print(f"     Max: {max(assistant_lengths)} chars")
    print(f"     Avg: {sum(assistant_lengths) / len(assistant_lengths):.0f} chars")
    
    # Token estimate (rough: chars / 4)
    avg_total_chars = (sum(user_lengths) + sum(assistant_lengths)) / len(dataset['train'])
    estimated_tokens = avg_total_chars / 4
    print(f"\n🎯 ESTIMATED TOKENS PER EXAMPLE: ~{estimated_tokens:.0f}")
    print(f"   (Rough estimate: total_chars / 4)")
    
    # Validation checks
    print(f"\n✅ VALIDATION CHECKS:")
    checks = {
        "All examples have 3 messages": all(len(ex['messages']) == 3 for ex in dataset['train']),
        "All messages have role+content": all(
            all('role' in msg and 'content' in msg for msg in ex['messages'])
            for ex in dataset['train']
        ),
        "Assistant responses are valid JSON": check_json_validity(dataset['train']),
        "No empty messages": all(
            all(len(msg['content']) > 0 for msg in ex['messages'])
            for ex in dataset['train']
        )
    }
    
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")
    
    print("\n" + "=" * 60)

def check_json_validity(examples) -> bool:
    """Check if all assistant responses are valid JSON"""
    try:
        for ex in examples:
            assistant_msg = ex['messages'][2]['content']
            json.loads(assistant_msg)
        return True
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description="Inspect training dataset")
    parser.add_argument("--dataset", default="hf_dataset", help="Path to HF dataset")
    
    args = parser.parse_args()
    
    inspect_training_data(args.dataset)

if __name__ == "__main__":
    main()
```

---

## 📋 PHASE 4: DOCUMENTATION UPDATES (30 minutes)

### Step 4.1: Update README.md

Add this section after "Running the Script" section:

```markdown
## Fine-Tuning for Production Deployment

This repository includes tools to prepare training data for fine-tuning a small open-source LLM to perform itemset extraction **without code execution**.

### Training Data Preparation Workflow

#### 1. Export Training Examples from Database
```powershell
python export_training_data.py --db runs.db --output training_data --model gpt_4_1
```

This creates:
- `training_data/all_training_examples.json` - Combined training examples
- `training_data/<dataset_id>_training.json` - Individual examples
- `training_data/export_summary.json` - Export metadata

#### 2. Create HuggingFace Dataset
```powershell
python create_hf_dataset.py --input training_data/all_training_examples.json --output hf_dataset
```

This creates:
- `hf_dataset/train/` - Training split (90% of data)
- `hf_dataset/validation/` - Validation split (10% of data)

#### 3. Inspect Dataset Quality
```powershell
python inspect_training_data.py --dataset hf_dataset
```

Shows:
- Split sizes
- Sample examples
- Token estimates
- Validation checks

### Fine-Tuning with HuggingFace + Claude

See `FINETUNING_PLAN_COMPREHENSIVE.md` for complete training guide including:
- Model selection (Qwen2.5-0.5B recommended)
- Training scripts (SFT with LoRA)
- HuggingFace Jobs submission
- Evaluation metrics

### Training Data Format

Each training example contains:
```json
{
  "messages": [
    {"role": "system", "content": "<extractor_system_prompt.md>"},
    {"role": "user", "content": "<CSV context + task description>"},
    {"role": "assistant", "content": "<ground truth itemsets (JSON array)>"}
  ],
  "metadata": {
    "dataset_id": "ds_0001",
    "itemset_count": 15,
    "min_support": 4
  }
}
```

Ground truth comes from validated Apriori outputs (100% validation pass rate across 300 runs).
```

### Step 4.2: Create Training Quick Start Guide
**File:** `TRAINING_QUICKSTART.md`

```markdown
# 🚀 Fine-Tuning Quick Start Guide

**Goal:** Train a small LLM to extract frequent itemsets from CSV data without running Apriori

---

## Prerequisites

✅ Python 3.10+ with virtual environment  
✅ 300 validated runs in `runs.db` (already completed)  
✅ HuggingFace account with API token  
✅ GPU access (Google Colab free tier or HF Jobs)  

---

## Step-by-Step Process

### 1. Install Dependencies (5 minutes)

```powershell
# Activate your virtual environment
.venv\Scripts\Activate.ps1

# Install HuggingFace dependencies
pip install datasets transformers trl peft accelerate bitsandbytes

# Verify installation
python -c "from datasets import Dataset; from transformers import AutoTokenizer; print('✅ Ready')"
```

### 2. Export Training Data (10 minutes)

```powershell
# Export 100 validated examples from runs.db
python export_training_data.py --model gpt_4_1

# Expected output:
#   ✅ Export Complete!
#   Examples: 100
#   Skipped: 0
#   Output: training_data\
```

### 3. Create HuggingFace Dataset (5 minutes)

```powershell
# Convert to HF format with train/val split
python create_hf_dataset.py

# Expected output:
#   ✅ HuggingFace Dataset Created!
#   Train: 90 examples
#   Validation: 10 examples
```

### 4. Inspect Dataset (2 minutes)

```powershell
# Validate format and check statistics
python inspect_training_data.py

# Expected checks:
#   ✅ All examples have 3 messages
#   ✅ Assistant responses are valid JSON
#   ✅ No empty messages
```

### 5. Submit Training Job with Claude

**Option A: Use HuggingFace Skills (Recommended)**

1. Install HF Skills in Claude Code:
   ```
   /plugin install hf-llm-trainer@huggingface-skills
   ```

2. Prompt Claude:
   ```
   Fine-tune Qwen2.5-0.5B-Instruct on my itemset extraction dataset.
   
   Dataset: hf_dataset/
   Task: Supervised fine-tuning (SFT)
   Hardware: t4-medium (should take ~30 min)
   
   Training config:
   - LoRA rank: 16
   - Learning rate: 2e-4
   - Epochs: 3
   - Batch size: 4
   - Max sequence length: 2048
   
   Push final model to Hub: <your-username>/itemset-extractor-qwen
   ```

**Option B: Local Training (Google Colab)**

Upload `hf_dataset/` to Google Drive, then use training script from `FINETUNING_PLAN_COMPREHENSIVE.md`.

---

## Expected Results

After training (~30-60 minutes):

✅ Fine-tuned model on HuggingFace Hub  
✅ Training metrics (loss curves, eval results)  
✅ Ready for inference testing  

### Next Steps

1. Test model on validation set
2. Compare F1 score against Apriori baseline
3. Quantize to GGUF for local deployment
4. Replace pipeline.py with trained model

---

## Troubleshooting

**Issue:** Export script fails  
**Solution:** Check runs.db has 100+ validated runs with `llm_model='gpt_4_1'`

**Issue:** Dataset creation fails  
**Solution:** Verify `extractor_system_prompt.md` exists in root directory

**Issue:** Claude can't access dataset  
**Solution:** Push dataset to HuggingFace Hub first:
```python
from datasets import load_from_disk
dataset = load_from_disk("hf_dataset")
dataset.push_to_hub("your-username/itemset-training-data")
```

---

## Cost Estimate

| Method | Hardware | Time | Cost |
|--------|----------|------|------|
| **HF Jobs (t4-medium)** | NVIDIA T4 | 30-60 min | $0.60 |
| **HF Jobs (a10g-small)** | NVIDIA A10G | 20-40 min | $1.10 |
| **Google Colab Free** | T4 (limited) | 1-2 hours | $0 |

**Recommended:** HF Jobs t4-medium for stability and reproducibility.
```

---

## 📦 PHASE 5: CREATE UTILITY SCRIPTS (1 hour)

### Step 5.1: Repository Status Checker
**File:** `check_repo_status.py`

```python
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
        conn = sqlite3.connect('runs.db')
        cursor = conn.cursor()
        
        # Total runs
        cursor.execute("SELECT COUNT(*) FROM runs")
        total_runs = cursor.fetchone()[0]
        checks['database_exists'] = True
        checks['total_runs'] = total_runs
        print(f"   ✅ Database found: {total_runs} total runs")
        
        # Runs by model
        cursor.execute("SELECT llm_model, COUNT(*) FROM runs GROUP BY llm_model")
        for model, count in cursor.fetchall():
            print(f"      - {model}: {count} runs")
        
        # Validated runs
        cursor.execute("SELECT COUNT(*) FROM runs WHERE validation_passed = 1")
        valid_runs = cursor.fetchone()[0]
        checks['valid_runs'] = valid_runs
        print(f"   ✅ Validated runs: {valid_runs}")
        
        conn.close()
    except Exception as e:
        checks['database_exists'] = False
        print(f"   ❌ Database error: {e}")
    
    # Check datasets
    print("\n📁 DATASETS:")
    datasets_dir = Path("datasets")
    if datasets_dir.exists():
        csv_files = list(datasets_dir.glob("ds_*.csv"))
        checks['dataset_count'] = len(csv_files)
        print(f"   ✅ Found {len(csv_files)} dataset files")
    else:
        checks['dataset_count'] = 0
        print(f"   ❌ datasets/ directory not found")
    
    # Check artifacts
    print("\n🗂️  ARTIFACTS:")
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        for subdir in ['apriori_outputs', 'extractor_outputs', 'validation_reports']:
            path = artifacts_dir / subdir
            if path.exists():
                count = len(list(path.glob("*.json")))
                print(f"   ✅ {subdir}: {count} files")
                checks[f'{subdir}_count'] = count
            else:
                print(f"   ⚠️  {subdir}: directory not found")
                checks[f'{subdir}_count'] = 0
    else:
        print(f"   ❌ artifacts/ directory not found")
    
    # Check system prompt
    print("\n📝 SYSTEM PROMPT:")
    prompt_file = Path("extractor_system_prompt.md")
    if prompt_file.exists():
        with open(prompt_file, 'r', encoding='utf-8') as f:
            lines = len(f.readlines())
        checks['system_prompt_exists'] = True
        print(f"   ✅ extractor_system_prompt.md: {lines} lines")
    else:
        checks['system_prompt_exists'] = False
        print(f"   ❌ extractor_system_prompt.md not found")
    
    # Check requirements
    print("\n📦 DEPENDENCIES:")
    req_file = Path("requirements.txt")
    if req_file.exists():
        with open(req_file, 'r') as f:
            reqs = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        checks['requirements_exists'] = True
        print(f"   ✅ requirements.txt: {len(reqs)} packages")
        
        # Check for HF packages
        hf_packages = ['datasets', 'transformers', 'trl', 'peft']
        missing = [pkg for pkg in hf_packages if not any(pkg in req for req in reqs)]
        if missing:
            print(f"   ⚠️  Missing HF packages: {', '.join(missing)}")
            checks['hf_packages_ready'] = False
        else:
            print(f"   ✅ All HF training packages present")
            checks['hf_packages_ready'] = True
    else:
        checks['requirements_exists'] = False
        print(f"   ❌ requirements.txt not found")
    
    # Overall status
    print("\n" + "=" * 60)
    print("📋 OVERALL STATUS:")
    
    ready = (
        checks.get('database_exists', False) and
        checks.get('valid_runs', 0) >= 100 and
        checks.get('dataset_count', 0) >= 100 and
        checks.get('system_prompt_exists', False)
    )
    
    if ready:
        print("   ✅ Repository READY for fine-tuning preparation!")
    else:
        print("   ⚠️  Repository needs attention before fine-tuning")
        if checks.get('valid_runs', 0) < 100:
            print("      - Need at least 100 validated runs")
        if checks.get('dataset_count', 0) < 100:
            print("      - Need at least 100 datasets")
        if not checks.get('system_prompt_exists', False):
            print("      - Missing system prompt file")
    
    print("=" * 60)
    
    return checks

if __name__ == "__main__":
    check_repo_status()
```

---

## ✅ EXECUTION CHECKLIST

### Pre-Cleanup Checklist
- [ ] Commit current work to git
- [ ] Verify runs.db has 300 validated runs
- [ ] Backup runs.db (copy to `runs_backup.db`)
- [ ] Review files to be archived/deleted

### Cleanup Execution (30 min)
- [ ] Create archive directories
- [ ] Move cleanup scripts to archive/
- [ ] Move legacy presentation files to archive/
- [ ] Remove ds_TEST.csv
- [ ] Clean __pycache__
- [ ] Update .gitignore
- [ ] Create azure.env.template
- [ ] Update requirements.txt with HF dependencies

### Script Creation (2 hours)
- [ ] Create export_training_data.py
- [ ] Create create_hf_dataset.py
- [ ] Create inspect_training_data.py
- [ ] Create check_repo_status.py

### Documentation Updates (30 min)
- [ ] Update README.md with fine-tuning section
- [ ] Create TRAINING_QUICKSTART.md
- [ ] Update .github/copilot-instructions.md

### Testing (30 min)
- [ ] Run check_repo_status.py
- [ ] Test export_training_data.py (1-2 examples)
- [ ] Test create_hf_dataset.py
- [ ] Test inspect_training_data.py
- [ ] Verify all scripts work end-to-end

### Final Steps
- [ ] Git commit all changes
- [ ] Push to GitHub
- [ ] Share repository with Claude for training assistance

---

## 🎯 ESTIMATED TOTAL TIME: 4-5 hours

---

## 📞 NEXT ACTIONS

After completing this cleanup:

1. **Run repository status check:**
   ```powershell
   python check_repo_status.py
   ```

2. **Export training data:**
   ```powershell
   python export_training_data.py
   ```

3. **Create HF dataset:**
   ```powershell
   python create_hf_dataset.py
   ```

4. **Share with Claude:**
   - Attach `hf_dataset/` directory
   - Attach `FINETUNING_PLAN_COMPREHENSIVE.md`
   - Attach `TRAINING_QUICKSTART.md`
   - Request custom training script generation

---

**Ready to begin? I can execute these steps systematically for you!**
