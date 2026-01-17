
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ===== 1. Load Dataset =====
DATASET_NAME = "OliverSlivka/itemset-extraction-v2"
print(f"💾 Loading dataset {DATASET_NAME} from Hugging Face Hub...")
dataset = load_dataset(DATASET_NAME)

# Create small subsets for the test run
train_dataset = dataset["train"].shuffle(seed=42).select(range(50))
eval_dataset = dataset["validation"].shuffle(seed=42)


print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval examples for test run.")
print(f"   Columns: {train_dataset.column_names}")
# The dataset should have a 'messages' column in ChatML format.
# SFTTrainer will automatically format it.

# ===== 2. Load Model with 4-bit Quantization =====
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # 3B model for better performance
OUTPUT_DIR = "OliverSlivka/qwen2.5-3b-itemset-test"  # Test repo on Hub

print(f"🔥 Loading {MODEL_NAME} with 4-bit quantization...")

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
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
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

print(f"🎯 LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}")

# ===== 4. Training Configuration for Test Run =====
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    push_to_hub=True,  # Push test model to verify everything works
    hub_strategy="end",  # Push only at the end

    # Training schedule for quick test
    num_train_epochs=1, # Single epoch is enough for a test
    per_device_train_batch_size=2,  # Smaller batch for 3B model
    gradient_accumulation_steps=8,  # Effective batch = 16
    learning_rate=2e-4,
    warmup_steps=5,
    max_steps=12,  # Limit steps for a quick run (50 examples / (2*8) batch size rounded up)

    # Optimization
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    gradient_checkpointing=True,

    # Precision
    fp16=True,
    bf16=False,  # Explicitly disable bfloat16 (T4 compatibility)

    # Logging
    logging_steps=1,

    # Evaluation
    eval_strategy="steps",
    eval_steps=5,

    # Saving
    save_strategy="no", # No need to save checkpoints for test

    # Sequence length
    max_length=2048,
)

print("✅ Training configuration set for test run")
print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   Max steps: {training_args.max_steps}")

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

# ===== 6. Train =====
print("\n🚀 Starting test training...")
print("="*60)

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

trainer.train()

print("="*60)
print("✅ Test training complete!")
print("\n🎉 Quick test run finished successfully!")
