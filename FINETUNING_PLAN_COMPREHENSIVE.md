# 🎯 COMPREHENSIVE FINE-TUNING PLAN
## Training Small Open-Source LLM for Frequent Itemset Extraction (Without Code Execution)

**Date:** December 14, 2025  
**Goal:** Fine-tune a small open-source LLM to identify and return frequent itemsets from CSV datasets **without running any code**

---

## 📊 CURRENT STATE ANALYSIS

### Your Repository Structure
```
✅ STRENGTHS:
• 300 high-quality runs (100 datasets × 3 models)
• Structured artifacts (Apriori outputs, LLM outputs, validation reports)
• SQLite database with comprehensive metadata
• Validated ground truth (100% validation pass rate)
• Diverse dataset dimensions (5-95 rows × 11-100 cols)
• Clean data pipeline with consistent formatting

📦 AVAILABLE DATA:
• Apriori outputs: Ground truth itemsets with count, rows, support
• LLM outputs: Model-generated itemsets with evidence
• Validation reports: Quality metrics and invariant checks
• Raw CSVs: Original transactional data (100 datasets)
• Database: Run metadata, timing, parameters
```

### Key Resources Identified from #file:resources

**1. UNSLOTH** (Optimal for your use case)
- 2x faster training, 60% less memory
- Supports Llama, Qwen, Mistral, Gemma models
- 500K context length support
- Perfect for small model fine-tuning (0.5B-3B params)
- **Location:** `resources/github/unsloth/`

**2. TRL (Transformer Reinforcement Learning)**
- SFT (Supervised Fine-Tuning) - Your primary method
- DPO (Direct Preference Optimization) - For refinement
- GRPO (Group Relative Policy Optimization) - Advanced RL
- **Location:** `resources/huggingface.co/docs/trl/`

**3. HF Skills Training Framework**
- End-to-end training workflow automation
- Dataset validation and formatting tools
- Cloud GPU training via Hugging Face Jobs
- **Location:** `resources/huggingface.co/skills-main/`

**4. Dataset Preparation Tools**
- `dataset_inspector.py` - Validates format compatibility
- `dataset_manager.py` - Creates HF datasets from your data
- **Location:** `resources/huggingface.co/skills-main/hf_dataset_creator/`

---

## 🎯 RECOMMENDED APPROACH

### Phase 1: Data Preparation & Formatting (Week 1)

#### Step 1.1: Create Training Dataset from Existing Runs

**Format:** SFT-style conversational format (required by TRL)

```python
# Target structure for each training example:
{
    "messages": [
        {
            "role": "system",
            "content": "You are a frequent itemset mining expert. Given CSV transaction data, identify all frequent itemsets with support count >= 4. Return ONLY a JSON array of itemsets with their counts and evidence rows. Do not execute code."
        },
        {
            "role": "user", 
            "content": "Dataset: ds_0001_5x53.csv\nTransactions (5 rows × 53 columns):\nRow 1: false, true, Location_27, ...\nRow 2: false, true, Product_A, ...\n...\n\nFind all frequent itemsets with minimum support count = 4."
        },
        {
            "role": "assistant",
            "content": "[{\"itemset\": [\"false\"], \"count\": 5, \"rows\": [\"Row 1\", \"Row 2\", \"Row 3\", \"Row 4\", \"Row 5\"]}, {\"itemset\": [\"true\"], \"count\": 5, \"rows\": [\"Row 1\", \"Row 2\", \"Row 3\", \"Row 4\", \"Row 5\"]}, {\"itemset\": [\"false\", \"true\"], \"count\": 5, \"rows\": [\"Row 1\", \"Row 2\", \"Row 3\", \"Row 4\", \"Row 5\"]}]"
        }
    ]
}
```

**Implementation Script:** `prepare_training_dataset.py`

```python
#!/usr/bin/env python3
"""
Converts runs.db + artifacts into HuggingFace dataset for SFT training.
Generates train/validation split with proper conversational format.
"""

import sqlite3
import json
from pathlib import Path
from datasets import Dataset
import pandas as pd

def load_csv_content(csv_path: str, max_rows: int = 20) -> str:
    """Load first N rows of CSV for context (avoid token overflow)"""
    df = pd.read_csv(csv_path)
    sample = df.head(max_rows)
    
    # Format as readable text
    rows_text = []
    for idx, row in sample.iterrows():
        items = [str(v) for v in row.values if pd.notna(v)]
        rows_text.append(f"Row {idx+1}: {', '.join(items)}")
    
    return f"Dataset: {Path(csv_path).name}\\nTransactions ({len(df)} rows × {len(df.columns)} columns):\\n" + "\\n".join(rows_text[:10]) + f"\\n... ({len(df)} total rows)"

def load_ground_truth(apriori_path: str) -> str:
    """Load Apriori output as ground truth response"""
    with open(apriori_path, 'r') as f:
        itemsets = json.load(f)
    
    # Convert to cleaner format (remove extra fields like support, size)
    cleaned = []
    for item in itemsets:
        cleaned.append({
            "itemset": item["itemset"],
            "count": item["count"],
            "rows": item["rows"]
        })
    
    return json.dumps(cleaned, ensure_ascii=False)

def create_training_examples():
    """Generate training examples from runs.db"""
    conn = sqlite3.connect('runs.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            dataset_name, 
            data_path,
            apriori_output_path,
            min_support,
            llm_model
        FROM runs 
        WHERE validation_passed = 1 
        AND llm_model = 'gpt_4_1'  -- Use only one model to avoid duplicates
    ''')
    
    examples = []
    system_prompt = (
        "You are a frequent itemset mining expert. Given CSV transaction data, "
        "identify all frequent itemsets with support count >= {min_support}. "
        "Return ONLY a JSON array of itemsets with their counts and evidence rows. "
        "Do not execute code. Inspect the data directly."
    )
    
    for row in cursor.fetchall():
        dataset_name, data_path, apriori_path, min_support, _ = row
        
        # Load CSV content (limited to avoid huge prompts)
        try:
            csv_content = load_csv_content(data_path, max_rows=20)
            ground_truth = load_ground_truth(apriori_path)
            
            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt.format(min_support=min_support)
                    },
                    {
                        "role": "user",
                        "content": f"{csv_content}\\n\\nFind all frequent itemsets with minimum support count = {min_support}."
                    },
                    {
                        "role": "assistant",
                        "content": ground_truth
                    }
                ]
            }
            examples.append(example)
        except Exception as e:
            print(f"Warning: Skipping {dataset_name}: {e}")
            continue
    
    conn.close()
    return examples

def main():
    print("🔄 Creating training dataset from runs.db...")
    
    examples = create_training_examples()
    print(f"✅ Generated {len(examples)} training examples")
    
    # Split into train/val (90/10)
    split_idx = int(len(examples) * 0.9)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    # Create HF datasets
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)
    
    print(f"📊 Train: {len(train_dataset)} examples")
    print(f"📊 Val: {len(val_dataset)} examples")
    
    # Save to disk
    train_dataset.save_to_disk("dataset/train")
    val_dataset.save_to_disk("dataset/val")
    
    # Push to Hub (optional)
    # train_dataset.push_to_hub("your-username/frequent-itemsets-train")
    # val_dataset.push_to_hub("your-username/frequent-itemsets-val")
    
    print("✅ Dataset saved to dataset/train and dataset/val")

if __name__ == "__main__":
    main()
```

**Expected Output:**
- 90 training examples (90% of 100 datasets)
- 10 validation examples (10% split)
- Each example = CSV context + ground truth itemsets

---

### Phase 2: Model Selection & Setup (Week 1)

#### Recommended Models (Small, Fast, Effective)

**Option 1: Qwen2.5-0.5B** ⭐ BEST FOR YOUR USE CASE
```
✅ Only 0.5B parameters (fits on consumer GPU)
✅ Excellent instruction following
✅ Fast inference (critical for production)
✅ Strong JSON generation capabilities
✅ Well-supported by Unsloth
```

**Option 2: Qwen2.5-1.5B**
```
✅ Slightly larger but more capable
✅ Better complex reasoning
✅ Still efficient for deployment
```

**Option 3: SmolLM2-1.7B**
```
✅ Recent model optimized for small tasks
✅ Good for structured outputs
✅ Lightweight deployment
```

#### Installation & Setup

```bash
# Install Unsloth (fastest training framework)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install TRL & dependencies
pip install trl transformers datasets peft accelerate bitsandbytes

# Verify installation
python -c "from unsloth import FastLanguageModel; print('✅ Unsloth ready')"
```

---

### Phase 3: Training Script (Week 2)

**File:** `train_itemset_extractor.py`

```python
#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-0.5B for frequent itemset extraction using Unsloth + TRL.
Based on best practices from HF Skills training framework.
"""

from unsloth import FastLanguageModel, is_bf16_supported
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import torch

# ============================================
# Configuration
# ============================================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "itemset-extractor-qwen-0.5b"
HUB_MODEL_ID = "your-username/itemset-extractor-qwen-0.5b"  # Change this

MAX_SEQ_LENGTH = 2048  # Adjust based on your CSV sizes
LORA_RANK = 16
LORA_ALPHA = 16

BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4  # Effective batch = 4*4 = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

# ============================================
# Load Model with Unsloth (2x faster)
# ============================================
print("🚀 Loading model with Unsloth...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect (fp16 or bf16)
    load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
)

# ============================================
# Apply LoRA (Low-Rank Adaptation)
# ============================================
print("⚙️ Applying LoRA adapters...")

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",  # Unsloth optimization
    random_state=42,
)

# ============================================
# Load Datasets
# ============================================
print("📦 Loading datasets...")

train_dataset = load_from_disk("dataset/train")
val_dataset = load_from_disk("dataset/val")

print(f"✅ Train: {len(train_dataset)} examples")
print(f"✅ Val: {len(val_dataset)} examples")

# ============================================
# Training Configuration
# ============================================
training_args = SFTConfig(
    # Output & Hub
    output_dir=OUTPUT_DIR,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
    hub_strategy="every_save",
    
    # Training params
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    
    # Optimization
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",  # Memory-efficient optimizer
    
    # Logging & Checkpointing
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=50,
    
    # Memory optimization
    fp16=not is_bf16_supported(),
    bf16=is_bf16_supported(),
    gradient_checkpointing=True,
    
    # Data
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,  # Don't pack multiple examples (our data has varying lengths)
    
    # Monitoring
    report_to="none",  # Change to "wandb" if you have Weights & Biases
)

# ============================================
# Initialize Trainer
# ============================================
print("🎯 Initializing trainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
)

# ============================================
# Train!
# ============================================
print("🚂 Starting training...")
print(f"   Model: {MODEL_NAME}")
print(f"   LoRA Rank: {LORA_RANK}")
print(f"   Batch Size: {BATCH_SIZE} × {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   LR: {LEARNING_RATE}")
print("="*60)

trainer.train()

# ============================================
# Save Final Model
# ============================================
print("💾 Saving model...")

# Save LoRA adapters
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Push to Hub
if training_args.push_to_hub:
    trainer.push_to_hub()
    print(f"✅ Model pushed to: https://huggingface.co/{HUB_MODEL_ID}")

print("🎉 Training complete!")
```

**Training Time Estimate:**
- **Hardware:** 1x T4 GPU (Google Colab free tier)
- **Time:** ~30-60 minutes for 3 epochs on 90 examples
- **Cost:** $0 (free tier) or ~$0.50 on Colab Pro

---

### Phase 4: Inference & Evaluation (Week 2)

**File:** `test_finetuned_model.py`

```python
#!/usr/bin/env python3
"""
Test fine-tuned itemset extractor on new CSV data.
Compare against Apriori ground truth.
"""

from unsloth import FastLanguageModel
import pandas as pd
import json

# ============================================
# Load Fine-Tuned Model
# ============================================
MODEL_PATH = "your-username/itemset-extractor-qwen-0.5b"  # From Hub or local

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# ============================================
# Prepare Test Input
# ============================================
def format_csv_for_inference(csv_path: str, max_rows: int = 20) -> str:
    """Format CSV into prompt"""
    df = pd.read_csv(csv_path)
    
    rows_text = []
    for idx, row in df.head(max_rows).iterrows():
        items = [str(v) for v in row.values if pd.notna(v)]
        rows_text.append(f"Row {idx+1}: {', '.join(items)}")
    
    return f"Dataset: {csv_path}\\nTransactions ({len(df)} rows × {len(df.columns)} columns):\\n" + "\\n".join(rows_text) + f"\\n... ({len(df)} total rows)"

# ============================================
# Run Inference
# ============================================
def extract_itemsets(csv_path: str, min_support: int = 4):
    """Extract itemsets using fine-tuned model"""
    
    csv_content = format_csv_for_inference(csv_path)
    
    messages = [
        {"role": "system", "content": f"You are a frequent itemset mining expert. Given CSV transaction data, identify all frequent itemsets with support count >= {min_support}. Return ONLY a JSON array of itemsets with their counts and evidence rows. Do not execute code."},
        {"role": "user", "content": f"{csv_content}\\n\\nFind all frequent itemsets with minimum support count = {min_support}."}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.1,  # Low temperature for deterministic output
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON from response
    try:
        # Find JSON array in response
        start = response.find('[')
        end = response.rfind(']') + 1
        json_str = response[start:end]
        itemsets = json.loads(json_str)
        return itemsets
    except:
        print("Warning: Failed to parse JSON from response")
        print("Raw response:", response)
        return []

# ============================================
# Test on Sample Dataset
# ============================================
if __name__ == "__main__":
    test_csv = "datasets/ds_0001_5x53.csv"
    
    print(f"🧪 Testing on: {test_csv}")
    print("="*60)
    
    itemsets = extract_itemsets(test_csv, min_support=4)
    
    print(f"\\n✅ Found {len(itemsets)} frequent itemsets:")
    print(json.dumps(itemsets, indent=2))
    
    # Compare with ground truth
    with open("artifacts/apriori_outputs/gpt_4_1_apriori_output_ds_0001_5x53_7a6d0b9e7fce.json") as f:
        ground_truth = json.load(f)
    
    print(f"\\n📊 Ground Truth: {len(ground_truth)} itemsets")
    print(f"📊 Model Output: {len(itemsets)} itemsets")
    
    # Calculate accuracy
    gt_itemsets = set(tuple(sorted(item["itemset"])) for item in ground_truth)
    pred_itemsets = set(tuple(sorted(item["itemset"])) for item in itemsets)
    
    correct = len(gt_itemsets & pred_itemsets)
    precision = correct / len(pred_itemsets) if pred_itemsets else 0
    recall = correct / len(gt_itemsets) if gt_itemsets else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\\n📈 Metrics:")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall: {recall:.2%}")
    print(f"   F1 Score: {f1:.2%}")
```

---

### Phase 5: Code Improvements for Better Training Data (Week 3)

#### Modification 1: Enhanced Artifact Outputs

**Update `pipeline.py` to include more training-friendly fields:**

```python
# Add to build_run_summary() function:

def build_run_summary_enhanced(apriori_sets, llm_sets, validation, args, 
                               validation_passed, dataset_meta, run_duration_ms,
                               apriori_path, llm_path, validation_path, 
                               summary_path, transactions, error_message=None):
    """Enhanced version with full transaction context for fine-tuning"""
    
    summary = build_run_summary(...)  # Call original function
    
    # ADD: Full transaction dump for training (limited to first N rows to avoid huge files)
    MAX_TRAINING_ROWS = 50
    summary["training_context"] = {
        "transactions_sample": transactions[:MAX_TRAINING_ROWS],
        "total_transaction_count": len(transactions),
        "sample_ratio": min(1.0, MAX_TRAINING_ROWS / len(transactions)),
        "formatted_prompt": format_training_prompt(transactions[:MAX_TRAINING_ROWS], args.min_support),
        "expected_response": json.dumps([
            {"itemset": item["itemset"], "count": item["count"], "rows": item["rows"]}
            for item in apriori_sets
        ])
    }
    
    return summary

def format_training_prompt(transactions, min_support):
    """Format transactions as training prompt"""
    rows_text = []
    for idx, items in enumerate(transactions[:20], 1):  # Limit to 20 for prompt
        rows_text.append(f"Row {idx}: {', '.join(items)}")
    
    return (
        f"Transactions ({len(transactions)} total rows):\\n" +
        "\\n".join(rows_text) +
        (f"\\n... ({len(transactions) - 20} more rows)" if len(transactions) > 20 else "") +
        f"\\n\\nFind all frequent itemsets with minimum support count = {min_support}."
    )
```

#### Modification 2: Add Dataset Difficulty Metadata

```python
# Add to compute_dataset_metadata():

def compute_dataset_metadata_enhanced(path, transactions):
    """Add difficulty indicators for stratified sampling"""
    meta = compute_dataset_metadata(path, transactions)
    
    # Calculate complexity metrics
    unique_items = set()
    for trans in transactions:
        unique_items.update(trans)
    
    avg_trans_length = sum(len(t) for t in transactions) / len(transactions)
    
    meta["complexity"] = {
        "unique_item_count": len(unique_items),
        "avg_transaction_length": avg_trans_length,
        "sparsity": 1.0 - (avg_trans_length / len(unique_items)) if unique_items else 0,
        "difficulty_score": len(unique_items) * avg_trans_length / len(transactions)
    }
    
    return meta
```

---

### Phase 6: Advanced Training Techniques (Optional, Week 4)

#### Option A: Multi-Stage Training

```python
# Stage 1: SFT on full dataset (baseline)
# Stage 2: DPO on preference pairs (quality refinement)

# Create preference pairs from validation data:
# - Chosen: LLM outputs that passed validation
# - Rejected: Synthetic errors (missing itemsets, wrong counts)
```

#### Option B: GRPO (Reinforcement Learning)

```python
# Use validation invariants as reward function
# Based on grpo_agent.py from resources

def itemset_reward(completions, ground_truth):
    """Reward based on validation metrics"""
    rewards = []
    for completion in completions:
        try:
            predicted = json.loads(completion)
            gt_sets = set(tuple(sorted(item["itemset"])) for item in ground_truth)
            pred_sets = set(tuple(sorted(item["itemset"])) for item in predicted)
            
            precision = len(gt_sets & pred_sets) / len(pred_sets) if pred_sets else 0
            recall = len(gt_sets & pred_sets) / len(gt_sets) if gt_sets else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            reward = f1 * 10  # Scale to 0-10 range
            
            # Penalize invalid JSON
            if not isinstance(predicted, list):
                reward -= 5
            
            # Bonus for exact match
            if gt_sets == pred_sets:
                reward += 2
            
            rewards.append(reward)
        except:
            rewards.append(-10)  # Severe penalty for invalid output
    
    return rewards
```

---

## 🚀 EXECUTION TIMELINE

### Week 1: Data Preparation
- [ ] Day 1-2: Run `prepare_training_dataset.py` to create HF dataset
- [ ] Day 3: Validate dataset format with `dataset_inspector.py`
- [ ] Day 4-5: Split train/val, push to Hub
- [ ] Day 6-7: Set up Unsloth environment, test model loading

### Week 2: Initial Training
- [ ] Day 1-2: Run first SFT training (3 epochs on Qwen2.5-0.5B)
- [ ] Day 3: Evaluate on validation set
- [ ] Day 4-5: Hyperparameter tuning (learning rate, LoRA rank)
- [ ] Day 6-7: Test inference on unseen datasets

### Week 3: Refinement
- [ ] Day 1-3: Implement pipeline improvements (enhanced artifacts)
- [ ] Day 4-5: Regenerate training data with improvements
- [ ] Day 6-7: Re-train with enhanced data

### Week 4: Advanced Techniques (Optional)
- [ ] Day 1-3: Implement DPO preference pairs
- [ ] Day 4-5: Train DPO refinement stage
- [ ] Day 6-7: Final evaluation and documentation

---

## 📋 CHECKLIST: Pre-Flight Verification

### Before Training:
- [ ] `runs.db` has 300 validated runs
- [ ] All artifact files exist and are valid JSON
- [ ] Training dataset created (90 train + 10 val examples)
- [ ] Hugging Face account created
- [ ] HF_TOKEN set with write permissions
- [ ] GPU access verified (Google Colab or local)
- [ ] Unsloth + TRL installed successfully

### During Training:
- [ ] Training loss decreasing steadily
- [ ] Validation loss not diverging (no overfitting)
- [ ] Sample outputs look reasonable (valid JSON)
- [ ] Checkpoints saving correctly
- [ ] Hub uploads working

### After Training:
- [ ] Model generates valid JSON arrays
- [ ] Itemsets match expected format
- [ ] Precision ≥ 80% on validation set
- [ ] Recall ≥ 80% on validation set
- [ ] Inference speed < 2 seconds per dataset
- [ ] Model size reasonable for deployment (< 2GB)

---

## 💡 KEY INSIGHTS FROM RESOURCES

### From Unsloth Documentation:
```
✅ Use FastLanguageModel.from_pretrained() for 2x speedup
✅ Enable 4-bit quantization (load_in_4bit=True) for memory efficiency
✅ Use gradient checkpointing ("unsloth" mode) for larger batches
✅ LoRA rank 16-32 is optimal for small models
✅ Max sequence length 2048-4096 for CSV data
```

### From TRL/HF Skills:
```
✅ Always include eval_dataset when using eval_strategy
✅ Use "messages" format for chat models
✅ Set packing=False for variable-length inputs
✅ Use adamw_8bit optimizer for memory savings
✅ Cosine scheduler with warmup_ratio=0.1 is stable
```

### From GRPO Agent Example:
```
✅ Reward functions should be simple and interpretable
✅ Penalize invalid outputs heavily (-10 reward)
✅ Bonus rewards for exact matches (+2)
✅ Scale rewards to 0-10 range for stability
```

---

## 🎯 SUCCESS CRITERIA

### Minimum Viable Model (MVP):
- [ ] Generates valid JSON 95% of the time
- [ ] F1 score ≥ 70% on validation set
- [ ] Inference < 5 seconds per dataset
- [ ] Works on datasets up to 50 rows

### Production-Ready Model:
- [ ] Generates valid JSON 99%+ of the time
- [ ] F1 score ≥ 85% on validation set
- [ ] Inference < 2 seconds per dataset
- [ ] Works on datasets up to 100 rows
- [ ] Handles edge cases (empty itemsets, all singletons)
- [ ] Robust to typos and formatting variations

---

## 📚 ADDITIONAL RESOURCES

### Required Reading:
1. `resources/huggingface.co/skills-main/hf-llm-trainer/SKILL.md` - Training workflow
2. `resources/github/unsloth/unsloth-main/README.md` - Unsloth setup
3. `resources/huggingface.co/docs/trl/sft_trainer.txt` - SFT documentation

### Code Templates:
1. `resources/huggingface.co/skills-main/hf-llm-trainer/scripts/train_sft_example.py`
2. `resources/github/grpo_agent.py` - Reward function examples
3. `resources/github/unsloth/unsloth-main/tests/saving/language_models/test_save_merged_grpo_model.py`

### Tools:
1. `dataset_inspector.py` - Validate dataset format
2. `dataset_manager.py` - Create HF datasets
3. `convert_to_gguf.py` - Deploy to local inference (Ollama, LM Studio)

---

## 🚨 CRITICAL WARNINGS

### DO NOT:
❌ Train without validation dataset (will hang)
❌ Use eval_strategy without eval_dataset
❌ Set timeout < expected training time (job fails, loses progress)
❌ Skip HF_TOKEN in job config (ephemeral env loses model)
❌ Use packing=True with variable-length inputs
❌ Ignore memory warnings (will OOM crash)
❌ Train on full CSV content (use max 20-50 rows for context)

### DO:
✅ Start with smallest model (0.5B) and scale up
✅ Monitor loss curves for overfitting
✅ Test inference frequently during training
✅ Save checkpoints every 50 steps
✅ Push to Hub regularly
✅ Validate JSON outputs programmatically
✅ Use temperature=0.1 for deterministic inference

---

## 🎓 LEARNING PATH

If you're new to fine-tuning:

**Day 1-2:** Read TRL SFT documentation
**Day 3-4:** Run example training script from HF Skills
**Day 5-7:** Adapt example to your data
**Week 2:** Run first training iteration
**Week 3+:** Iterate based on results

---

## 💰 COST ESTIMATES

### Free Tier (Google Colab):
- **Cost:** $0
- **Time:** ~1-2 hours per training run
- **Limitations:** 12-hour session limit, T4 GPU only
- **Best for:** Prototyping, small models (0.5B-1.5B)

### Colab Pro ($10/month):
- **Cost:** $10/month + ~$0.50 per training run
- **Time:** ~30-60 minutes per training run
- **GPU:** A100 or V100 available
- **Best for:** Production training, larger models (3B-7B)

### Hugging Face Jobs:
- **Cost:** ~$0.30-1.00 per training run
- **Time:** ~20-40 minutes per training run
- **GPU:** T4, A10G, A100 available
- **Best for:** Automated training pipelines

---

## 📊 EXPECTED RESULTS

### After Initial SFT (Week 2):
```
Precision: 60-75%
Recall: 65-80%
F1 Score: 62-77%
Valid JSON: 85-95%
```

### After Refinement (Week 3):
```
Precision: 75-85%
Recall: 80-90%
F1 Score: 77-87%
Valid JSON: 95-98%
```

### After DPO/GRPO (Week 4):
```
Precision: 85-95%
Recall: 90-95%
F1 Score: 87-95%
Valid JSON: 98-99%
```

---

## 🔧 TROUBLESHOOTING

### Issue: Model generates invalid JSON
**Solution:** Lower temperature to 0.0, add JSON validation to reward function

### Issue: Model misses small itemsets
**Solution:** Emphasize singletons in training data, add examples with count=4 (minimum)

### Issue: Training loss not decreasing
**Solution:** Increase learning rate to 5e-4, reduce LoRA rank to 8, check data quality

### Issue: Out of memory errors
**Solution:** Reduce batch size to 2, enable gradient_checkpointing, use 4-bit quantization

### Issue: Model overfits (val loss increases)
**Solution:** Add more diverse datasets, reduce epochs to 2, increase dropout to 0.1

---

## 📞 NEXT STEPS

1. **Run `prepare_training_dataset.py`** to create your training data
2. **Verify dataset format** with `dataset_inspector.py`
3. **Set up Unsloth** in Google Colab or local environment
4. **Run first training** with `train_itemset_extractor.py`
5. **Evaluate results** with `test_finetuned_model.py`
6. **Iterate** based on metrics

**Ready to start? I can help you implement any of these scripts or troubleshoot issues along the way!**

---

Generated: December 14, 2025  
Version: 1.0
