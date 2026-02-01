"""
Simplified SFT training script for Qwen2.5-0.5B-Instruct
Based on official HuggingFace TRL examples
Dataset loaded from GitHub to avoid Hub caching issues
"""

import subprocess
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ===== 1. Clone Dataset from GitHub =====
GIT_TOKEN = "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"
GIT_REPO_URL = f"https://{GIT_TOKEN}@github.com/oliversl1vka/itemsety-qwen-finetuning.git"
CLONE_PATH = "/tmp/itemsety-qwen-finetuning"
DATASET_PATH = f"{CLONE_PATH}/hf_dataset_enhanced"

print("📦 Cloning dataset from private GitHub repo...")
subprocess.run(['git', 'clone', GIT_REPO_URL, CLONE_PATH], check=True)
print("✅ Clone complete")

# Security: Remove .git to avoid token exposure
subprocess.run(['rm', '-rf', f"{CLONE_PATH}/.git"], check=True)
print("🔐 Removed .git directory")

# ===== 2. Load Dataset =====
print(f"💾 Loading dataset from {DATASET_PATH}...")
dataset = load_from_disk(DATASET_PATH)
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval examples")
print(f"   Columns: {train_dataset.column_names}")
print(f"   First example keys: {list(train_dataset[0].keys())}")

# ===== 3. Load Model with 4-bit Quantization =====
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "OliverSlivka/qwen-itemsety-qlora"

print(f"🔥 Loading {MODEL_NAME} with 4-bit quantization...")

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✅ Model and tokenizer loaded with 4-bit quantization")

# ===== 4. LoRA Configuration =====
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

print(f"🎯 LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}")

# ===== 5. Training Configuration =====
training_args = SFTConfig(
    # Output & Hub
    output_dir=OUTPUT_DIR,
    push_to_hub=True,
    hub_model_id=OUTPUT_DIR,
    
    # Training schedule
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=10,
    max_steps=-1,  # Train for full epochs
    
    # Optimization
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    gradient_checkpointing=True,
    
    # Precision
    bf16=True,
    
    # Logging
    logging_steps=5,
    report_to="trackio",
    trackio_space_id=OUTPUT_DIR,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=20,
    
    # Saving
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    
    # Sequence length
    max_length=2048,
)

print("✅ Training configuration set")
print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Learning rate: {training_args.learning_rate}")

# ===== 6. Initialize Trainer =====
print("🎯 Initializing SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)

print("✅ Trainer initialized")

# Show GPU memory before training
if torch.cuda.is_available():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    max_memory = round(gpu_stats.total_memory / 1024**3, 3)
    print(f"\n🖥️  GPU: {gpu_stats.name}")
    print(f"   Max memory: {max_memory} GB")
    print(f"   Reserved: {start_memory} GB")

# ===== 7. Train =====
print("\n🚀 Starting training...")
print("="*60)

trainer_stats = trainer.train()

print("="*60)
print("✅ Training complete!")

# Show final stats
if torch.cuda.is_available():
    used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    training_memory = round(used_memory - start_memory, 3)
    print(f"\n📊 Training stats:")
    print(f"   Runtime: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
    print(f"   Peak memory: {used_memory} GB ({round(used_memory/max_memory*100, 1)}%)")
    print(f"   Training memory: {training_memory} GB")

# ===== 8. Push to Hub =====
print("\n💾 Pushing final model to Hub...")
trainer.push_to_hub()
print(f"✅ Model pushed to: https://huggingface.co/{OUTPUT_DIR}")
print(f"📊 View training metrics at: https://huggingface.co/spaces/{OUTPUT_DIR}")

print("\n🎉 All done!")
