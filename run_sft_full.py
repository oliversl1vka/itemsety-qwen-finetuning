#!/usr/bin/env python3
"""
PRODUCTION Fine-Tuning Script for Qwen2.5-3B on Itemset Extraction
Full training on 439 examples, 3 epochs, push to Hub
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ===== 1. Load Full Dataset =====
DATASET_NAME = "OliverSlivka/itemset-extraction-v2"
print(f"💾 Loading full dataset {DATASET_NAME} from Hugging Face Hub...")
dataset = load_dataset(DATASET_NAME)

# Use FULL training and validation sets
train_dataset = dataset["train"]  # 439 examples
eval_dataset = dataset["validation"]  # 49 examples

print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval examples.")
print(f"   Columns: {train_dataset.column_names}")

# ===== 2. Load Model with 4-bit Quantization =====
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # Larger model for better performance
OUTPUT_DIR = "OliverSlivka/qwen2.5-3b-itemset-extractor"  # Hub repo

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

# ===== 3. LoRA Configuration =====
peft_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  # LoRA alpha
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

print(f"🎯 LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}")

# ===== 4. Training Configuration for PRODUCTION =====
# Calculate steps: 439 examples / (4 batch * 4 gradient_accum) = ~27 steps per epoch
# 3 epochs = ~81 steps total
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    push_to_hub=True,  # Push final model to Hub
    hub_strategy="end",  # Push only at end
    
    # Training schedule
    num_train_epochs=3,  # Full 3 epochs
    per_device_train_batch_size=2,  # Smaller batch for 3B model
    gradient_accumulation_steps=8,  # Effective batch = 16
    learning_rate=2e-4,
    warmup_steps=10,
    max_steps=-1,  # Use epochs instead of steps
    
    # Optimization
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    gradient_checkpointing=True,
    
    # Precision
    fp16=True,  # Use FP16 for training
    
    # Logging
    logging_steps=5,
    logging_first_step=True,
    report_to="none",  # No W&B/TensorBoard
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=20,
    
    # Saving
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,  # Keep only 2 best checkpoints
    
    # Sequence length
    max_length=2048,
)

print("✅ Training configuration set for PRODUCTION")
print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Estimated steps: ~{len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")

# ===== 5. Initialize Trainer =====
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
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    max_memory = round(gpu_stats.total_memory / 1024**3, 3)
    print(f"\n🖥️  GPU: {gpu_stats.name}")
    print(f"   Max memory: {max_memory} GB")
    print(f"   Reserved: {start_memory} GB")
else:
    print("\n⚠️  No GPU detected! Training will be VERY slow on CPU.")
    start_memory = 0

# ===== 6. Train =====
print("\n🚀 Starting PRODUCTION training...")
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
    print(f"   Samples/second: {round(trainer_stats.metrics['train_samples_per_second'], 2)}")
    print(f"   Peak memory: {used_memory} GB ({round(used_memory/max_memory*100, 1)}%)")
    print(f"   Training memory: {training_memory} GB")

# ===== 7. Push to Hub =====
print(f"\n💾 Pushing final model to {OUTPUT_DIR}...")
trainer.push_to_hub()
print(f"✅ Model pushed to: https://huggingface.co/{OUTPUT_DIR}")

print("\n🎉 Production training complete!")
print(f"\nYour model is ready at: https://huggingface.co/{OUTPUT_DIR}")
