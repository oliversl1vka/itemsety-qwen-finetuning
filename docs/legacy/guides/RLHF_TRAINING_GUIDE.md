# RLHF Training for Itemset Extraction

This directory contains scripts for **Reinforcement Learning from Human Feedback (RLHF)** to train Qwen models that extract frequent itemsets without using the Apriori algorithm.

## 🎯 Why RLHF Instead of SFT?

**Supervised Fine-Tuning (SFT)** trains on correct answers only, but models can still produce:
- **Hallucinations**: Inventing non-existent itemsets
- **Wrong counts**: Incorrect support values
- **Missing itemsets**: Failing to find valid patterns
- **Format errors**: Malformed JSON outputs

**RLHF** teaches the model to **prefer correct answers over common mistakes** by training on preference pairs:
- ✅ **Chosen**: Apriori ground truth (high quality)
- ❌ **Rejected**: Synthetic errors (low quality)

This makes the model more robust and aligned with human preferences for **correctness, completeness, and format compliance**.

---

## 📚 Background: RLHF Methods

Based on [awesome-RLHF](https://github.com/opendilab/awesome-RLHF), there are three main approaches:

### 1. **DPO (Direct Preference Optimization)** ⭐ RECOMMENDED
- **Simplest**: No reward model needed
- **Fastest**: Single training phase
- **Effective**: State-of-the-art results
- **Paper**: [DPO: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- **Use case**: When you have preference pairs (chosen/rejected)

### 2. **PPO (Proximal Policy Optimization)**
- **Classic RLHF**: Used by InstructGPT, ChatGPT
- **Complex**: Requires training a reward model first
- **Slower**: Two-stage training process
- **Use case**: When you need fine-grained control over rewards

### 3. **RLAIF (RL from AI Feedback)**
- **Variation**: Uses AI judge instead of human labels
- **Scalable**: Can generate unlimited preference pairs
- **Use case**: When human labeling is expensive

**We use DPO** because it's simpler, faster, and equally effective for our task.

---

## 🔄 RLHF Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Generate Ground Truth (Apriori + GPT-4)                  │
│    python pipeline.py --data-dir data/datasets_v2 --llm-full│
│    → Creates validated runs in runs.db                       │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Export RLHF Preference Pairs                             │
│    python src/training/export_rlhf_training_data.py         │
│    → Creates chosen/rejected pairs with error types          │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Create HuggingFace Dataset (DPO format)                  │
│    python src/training/create_rlhf_hf_dataset.py --format dpo│
│    → Converts to HF dataset with chat templates              │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Train with DPO                                            │
│    python src/training/run_dpo_training.py                   │
│    → Fine-tunes Qwen2.5-3B with preference optimization      │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Evaluate                                                  │
│    python src/evaluation/eval_finetuned_model.py             │
│    → Test on unseen datasets, compute metrics                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install transformers datasets trl peft bitsandbytes accelerate wandb

# Ensure you have:
# - runs.db with validated pipeline runs
# - data/datasets_v2/ with CSV files
# - extractor_system_prompt.md
```

### Step 1: Export RLHF Data

```bash
python src/training/export_rlhf_training_data.py \
    --db runs.db \
    --output data/rlhf_training_v1 \
    --model gpt_4_1 \
    --num-rejected 3
```

**Output:**
- `data/rlhf_training_v1/all_rlhf_pairs.json` - All preference pairs
- `data/rlhf_training_v1/ds_XXXX_rlhf.json` - Per-dataset pairs
- `data/rlhf_training_v1/rlhf_export_summary.json` - Statistics

**What it does:**
- Loads validated Apriori runs from `runs.db`
- Creates 3 rejected variants per dataset:
  1. **Hallucinations**: Adds fake itemsets
  2. **Missing itemsets**: Removes valid patterns
  3. **Wrong counts**: Corrupts support values
- Formats as preference pairs: `{prompt, chosen, rejected, error_type}`

### Step 2: Create HuggingFace Dataset

```bash
python src/training/create_rlhf_hf_dataset.py \
    --input data/rlhf_training_v1/all_rlhf_pairs.json \
    --output data/hf_rlhf_dataset_v1 \
    --format dpo \
    --train-split 0.9
```

**Output:**
- `data/hf_rlhf_dataset_v1/train/` - Training split
- `data/hf_rlhf_dataset_v1/validation/` - Validation split
- `data/hf_rlhf_dataset_v1/dataset_metadata.json` - Statistics

**Format options:**
- `--format dpo` - Direct Preference Optimization (recommended)
- `--format ppo` - PPO reward modeling format
- `--format conversational` - TRL conversational format

### Step 3: Train with DPO

```bash
python src/training/run_dpo_training.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --dataset_path data/hf_rlhf_dataset_v1 \
    --output_dir ./dpo_checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --beta 0.1 \
    --use_4bit \
    --use_lora
```

**Key parameters:**
- `--beta 0.1` - DPO temperature (higher = stronger preference)
- `--use_lora` - Use LoRA for efficient training
- `--use_4bit` - 4-bit quantization (saves memory)

**Training time:**
- ~60-90 minutes for 500 datasets × 3 variants = 1500 pairs
- With 4-bit + LoRA: ~8 GB VRAM

### Step 4: Evaluate

```bash
python src/evaluation/eval_finetuned_model.py \
    --model-path ./dpo_checkpoints/final_model \
    --eval-dir eval_datasets \
    --count 10
```

---

## 📊 Error Types in RLHF Data

The rejected responses contain 6 types of common mistakes:

| Error Type | Description | Example |
|------------|-------------|---------|
| **hallucination** | Adds fake itemsets not in data | Invents `["milk", "eggs"]` when it doesn't exist |
| **missing_itemsets** | Removes 20-40% of valid itemsets | Finds 5 itemsets instead of 8 |
| **wrong_counts** | Corrupts support counts by ±1-5 | Reports count=5 instead of count=7 |
| **wrong_evidence** | Incorrect row references | Claims `["Row 1", "Row 5"]` when should be `["Row 2", "Row 8"]` |
| **subset_superset_confusion** | Includes redundant sets | Returns both `["A", "B"]` and `["A", "B", "C"]` |
| **below_min_support** | Includes itemsets below threshold | Returns itemset with count=2 when min_support=3 |

---

## 🔬 DPO Training Details

### What is DPO?

Direct Preference Optimization trains the model to maximize the likelihood ratio:

```
DPO Loss = -log(σ(β * log(π_θ(y_w | x) / π_ref(y_w | x)) - β * log(π_θ(y_l | x) / π_ref(y_l | x))))
```

Where:
- `π_θ` = Policy model (being trained)
- `π_ref` = Reference model (frozen copy of base model)
- `y_w` = Chosen response (ground truth)
- `y_l` = Rejected response (synthetic error)
- `β` = Temperature parameter (controls strength)
- `σ` = Sigmoid function

**Intuition**: Train the model to assign higher probability to chosen responses than rejected ones, relative to the reference model.

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | 0.1 | Temperature (0.1-0.5: lower = subtle, higher = aggressive) |
| `learning_rate` | 5e-5 | Lower than SFT (typically 5e-6 to 1e-4) |
| `num_train_epochs` | 3 | DPO converges faster than SFT |
| `max_length` | 2048 | Max sequence length (prompt + response) |
| `lora_r` | 64 | LoRA rank (16-128) |
| `lora_alpha` | 16 | LoRA scaling (usually r/4) |

### Memory Requirements

| Configuration | VRAM | Speed | Quality |
|---------------|------|-------|---------|
| **4-bit + LoRA** | ~8 GB | Fast | Good ⭐ |
| 8-bit + LoRA | ~12 GB | Medium | Better |
| Full precision | ~24 GB | Slow | Best |

---

## 📈 Expected Results

### Comparison: SFT vs DPO

| Metric | SFT Baseline | DPO (Expected) | Improvement |
|--------|--------------|----------------|-------------|
| F1 Score | 0.75 | 0.82 | +9% |
| Exact Match | 0.45 | 0.55 | +22% |
| Hallucination Rate | 8% | 3% | -63% |
| JSON Parse Rate | 92% | 98% | +7% |

### Why DPO Works Better

1. **Explicit error modeling**: Learns what NOT to do
2. **Preference alignment**: Optimizes for human-judged quality
3. **Robustness**: Better generalization to unseen data
4. **Format compliance**: Stronger adherence to JSON structure

---

## 🛠️ Advanced Usage

### Custom Error Types

Edit `export_rlhf_training_data.py` to add custom error generators:

```python
def generate_custom_error(ground_truth, min_support):
    """Your custom error logic"""
    # Example: Swap item names
    corrupted = copy.deepcopy(ground_truth)
    for item in corrupted:
        if random.random() < 0.3:
            # Swap two items in itemset
            if len(item["itemset"]) >= 2:
                i, j = random.sample(range(len(item["itemset"])), 2)
                item["itemset"][i], item["itemset"][j] = item["itemset"][j], item["itemset"][i]
    return corrupted, "item_swap"
```

### Multi-Objective DPO

Train on multiple criteria simultaneously:

```python
# Add reward annotations to rejected responses
rejected_with_scores = {
    "response": rejected_response,
    "correctness_score": 0.3,  # Low correctness
    "format_score": 0.8,       # Good format
    "completeness_score": 0.5, # Missing some
}
```

### Ensemble Training

Train multiple DPO models with different betas:

```bash
# Conservative (subtle corrections)
python run_dpo_training.py --beta 0.05 --output_dir dpo_beta005

# Moderate (balanced)
python run_dpo_training.py --beta 0.1 --output_dir dpo_beta010

# Aggressive (strong preferences)
python run_dpo_training.py --beta 0.3 --output_dir dpo_beta030
```

Then ensemble at inference time.

---

## 🐛 Troubleshooting

### Issue: OOM during training

**Solution:**
```bash
# Reduce batch size
--per_device_train_batch_size 1 --gradient_accumulation_steps 16

# Use smaller model
--model_name Qwen/Qwen2.5-1.5B-Instruct

# Reduce max length
--max_length 1024 --max_prompt_length 512
```

### Issue: Model ignores preferences

**Symptoms**: DPO loss decreases but validation metrics don't improve

**Solution:**
```bash
# Increase beta (stronger preference signal)
--beta 0.3

# Increase learning rate
--learning_rate 1e-4

# More epochs
--num_train_epochs 5
```

### Issue: Overfitting on error types

**Symptoms**: Perfect on training set, poor on validation

**Solution:**
```bash
# More error diversity
--num-rejected 5  # Instead of 3

# Regularization
--lora_dropout 0.1  # Increase dropout

# Data augmentation
# Add more unique datasets to runs.db
```

---

## 📚 References

### Papers
- [DPO: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)

### Codebases
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [awesome-RLHF](https://github.com/opendilab/awesome-RLHF)
- [DPO Reference Implementation](https://github.com/eric-mitchell/direct-preference-optimization)

### Datasets
- [HH-RLHF (Anthropic)](https://github.com/anthropics/hh-rlhf)
- [Stanford SHP](https://huggingface.co/datasets/stanfordnlp/SHP)
- [OpenAI WebGPT Comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons)

---

## 📝 TODO

- [ ] Add PPO training script (for comparison)
- [ ] Implement reward model training
- [ ] Add RLAIF variant (use Qwen-32B as judge)
- [ ] Multi-objective DPO with weighted preferences
- [ ] Online RLHF (iterative data collection)
- [ ] A/B testing framework for model comparison

---

**Last Updated:** 2026-02-03  
**Maintained By:** Oliver Slivka  
**License:** Apache 2.0
