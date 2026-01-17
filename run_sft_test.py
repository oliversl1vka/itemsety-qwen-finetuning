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

# ===== 2. Load Model with 4-bit Quantization =====
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "OliverSlivka/qwen2.5-3b-itemset-test"

print(f"🔥 Loading {MODEL_NAME} with 4-bit quantization...")

# 4-bit quantization config - use float32 for compute to avoid bf16 issues
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,  # Use fp32 for computation (T4 safe)
    bnb_4bit_use_double_quant=True,
)

# Load model - explicitly use float32 to avoid bf16 issues on T4
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",  # Avoid flash attention which might use bf16
)

# Ensure model is in float32 for non-quantized parts
for param in model.parameters():
    if param.dtype == torch.bfloat16:
        param.data = param.data.to(torch.float32)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✅ Model and tokenizer loaded with 4-bit quantization (fp32 compute)")

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
# CRITICAL: Disable ALL mixed precision to avoid bf16 issues on T4
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    push_to_hub=True,
    hub_strategy="end",

    # Training schedule
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Reduced from 2 to avoid OOM
    gradient_accumulation_steps=16,  # Increased to maintain effective batch
    learning_rate=2e-4,
    warmup_steps=5,
    max_steps=12,

    # Optimization
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    gradient_checkpointing=True,

    # CRITICAL: Disable ALL mixed precision (fp16 AND bf16)
    # This avoids the GradScaler bf16 issue on T4
    fp16=False,
    bf16=False,

    # Logging
    logging_steps=1,
    report_to="none",

    # Evaluation - DISABLED to avoid OOM during eval
    eval_strategy="no",
    # eval_steps=5,  # Disabled

    # Saving
    save_strategy="no",

    # Sequence length - reduced to save memory
    max_length=1024,
    
    # Memory optimization
    per_device_eval_batch_size=1,
)

print("✅ Training configuration set for test run (fp32 mode - T4 safe)")
print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   Max steps: {training_args.max_steps}")
print(f"   Mixed precision: DISABLED (fp32 training)")

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

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Check model dtype
print(f"Model dtype check:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.dtype}")
        break

trainer.train()

print("="*60)
print("✅ Test training complete!")
print("\n🎉 Quick test run finished successfully!")
