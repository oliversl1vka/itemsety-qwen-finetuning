# Qwen2.5-0.5B Fine-Tuning for Frequent Itemset Extraction

Fine-tuning Qwen2.5-0.5B to extract frequent itemsets from CSV datasets without using Apriori algorithm.

## Quick Start

### 1. Setup Environment

```powershell
# Clone repository
git clone https://github.com/oliversl1vka/itemsety-qwen-finetuning.git
cd itemsety-qwen-finetuning

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install datasets transformers trl peft accelerate torch pandas
```

### 2. Prepare Dataset

```powershell
# Export training data from runs.db (already done - 98 examples)
python export_training_data.py

# Create HuggingFace dataset
python create_hf_dataset.py
```

**Dataset Statistics:**
- **Total examples**: 98 (88 train / 10 validation)
- **Format**: Conversational (system + user + assistant)
- **Average itemsets per example**: ~14.5
- **Model**: gpt_4_1 (ground truth)

### 3. Train Model

#### Option A: Local Training (CPU/GPU)

```powershell
# Basic training (16-bit, LoRA)
python train_qwen_sft.py --epochs 3 --batch-size 2

# With 4-bit quantization (QLoRA) - requires less memory
python train_qwen_sft.py --epochs 3 --batch-size 2 --use-4bit

# Custom parameters
python train_qwen_sft.py \
  --epochs 5 \
  --batch-size 4 \
  --gradient-accumulation 2 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --max-seq-length 2048
```

**Training Arguments:**
- `--model-name`: Base model (default: `Qwen/Qwen2.5-0.5B-Instruct`)
- `--dataset-path`: HF dataset path (default: `hf_dataset`)
- `--output-dir`: Output directory (default: `qwen_itemset_model`)
- `--epochs`: Number of epochs (default: 3)
- `--batch-size`: Training batch size (default: 2)
- `--gradient-accumulation`: Gradient accumulation steps (default: 4)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--use-4bit`: Enable 4-bit quantization (QLoRA)
- `--lora-r`: LoRA rank (default: 16)
- `--lora-alpha`: LoRA alpha (default: 32)
- `--max-seq-length`: Maximum sequence length (default: 2048)

#### Option B: HuggingFace Jobs (Recommended)

HuggingFace Jobs provides free GPU compute for training:

1. **Push to GitHub** (already done):
   ```powershell
   git add .
   git commit -m "feat: Add Qwen2.5 fine-tuning scripts"
   git push origin main
   ```

2. **Create HuggingFace Space/Job**:
   - Go to: https://huggingface.co/spaces
   - Create new Space with GPU (e.g., T4 medium)
   - Link GitHub repo: `oliversl1vka/itemsety-qwen-finetuning`
   - Run training script

3. **Submit Training Job**:
   ```python
   # See FINETUNING_PLAN_COMPREHENSIVE.md for detailed HF Jobs setup
   ```

### 4. Evaluate Model

```powershell
# Run evaluation against ground truth
python evaluate_model.py --model-path qwen_itemset_model/final
```

### 5. Use Fine-Tuned Model

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Load model
model = AutoPeftModelForCausalLM.from_pretrained("qwen_itemset_model/final")
tokenizer = AutoTokenizer.from_pretrained("qwen_itemset_model/final")

# Prepare input
messages = [
    {"role": "system", "content": open("extractor_system_prompt.md").read()},
    {"role": "user", "content": "CSV_DATA_HERE\n\nFind all frequent itemsets with minimum support count = 3."}
]

# Generate
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
outputs = model.generate(inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Model Architecture

- **Base Model**: Qwen2.5-0.5B-Instruct
- **Parameters**: 494M total, ~8.8M trainable (1.75% with LoRA)
- **Fine-Tuning Method**: LoRA (Low-Rank Adaptation)
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **LoRA Config**: r=16, alpha=32, dropout=0.05

## Training Details

- **Optimizer**: AdamW (8-bit paged for QLoRA)
- **Learning Rate**: 2e-4 with cosine schedule
- **Warmup**: 10% of total steps
- **Gradient Checkpointing**: Enabled
- **Mixed Precision**: BFloat16
- **Batch Size**: 2 per device (effective: 8 with grad accumulation)
- **Max Sequence Length**: 2048 tokens

## Repository Structure

```
itemsety-qwen-finetuning/
├── train_qwen_sft.py           # Main training script
├── create_hf_dataset.py        # Dataset preparation
├── export_training_data.py     # Export from runs.db
├── inspect_training_data.py    # Dataset inspection
├── extractor_system_prompt.md  # System prompt template
├── pipeline.py                 # Apriori + LLM pipeline
├── dataset_generation.py       # Synthetic dataset generator
├── visualization.py            # Result visualization
├── requirements.txt            # Python dependencies
├── hf_dataset/                 # HuggingFace dataset (88 train, 10 val)
├── training_data/              # Exported examples (gitignored)
├── runs.db                     # SQLite database (gitignored)
└── datasets/                   # CSV datasets (gitignored)
```

## Security Notes

- ✅ `.gitignore` protects all `.env` files
- ✅ `azure.env` is **NOT** in repository
- ✅ `azure.env.template` provided as safe template
- ✅ Repository is **PRIVATE** on GitHub
- ✅ Training data, datasets, and database are gitignored

## Next Steps

1. **Run Full Training**: `python train_qwen_sft.py --epochs 5 --use-4bit`
2. **Evaluate**: Compare fine-tuned model vs gpt_4_1 ground truth
3. **Optimize**: Tune LoRA rank, learning rate, batch size
4. **Deploy**: Push model to HuggingFace Hub
5. **Integrate**: Replace Azure OpenAI with fine-tuned model in pipeline

## Resources

- [TRAINING_QUICKSTART.md](TRAINING_QUICKSTART.md) - Step-by-step guide
- [FINETUNING_PLAN_COMPREHENSIVE.md](FINETUNING_PLAN_COMPREHENSIVE.md) - Detailed plan
- [extractor_system_prompt.md](extractor_system_prompt.md) - System prompt
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)

## License

MIT License - See LICENSE file for details.

## Citation

If you use this work, please cite:

```bibtex
@software{itemsety_qwen_finetuning,
  title={Frequent Itemset Extraction with Qwen2.5-0.5B},
  author={Oliver Slivka},
  year={2025},
  url={https://github.com/oliversl1vka/itemsety-qwen-finetuning}
}
```
