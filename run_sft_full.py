import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ===== 1. Load Dataset =====
DATASET_NAME = "OliverSlivka/itemset-extraction-v2"
print(f"💾 Loading dataset {DATASET_NAME} from Hugging Face Hub...")
dataset = load_dataset(DATASET_NAME)

# Use full training set for production
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval examples.")
print(f"   Columns: {train_dataset.column_names}")

# ===== 2. Load Model with 4-bit Quantization =====
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "OliverSlivka/qwen2.5-3b-itemset-extractor"

print(f"🔥 Loading {MODEL_NAME} with 4-bit quantization...")

# 4-bit quantization config - use float32 for compute to avoid bf16 issues on T4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,  # Use fp32 for computation (T4 safe)
    bnb_4bit_use_double_quant=True,
)

# Load model - use eager attention to avoid bf16 issues
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

# ===== 4. Training Configuration for PRODUCTION =====
# CRITICAL: Disable ALL mixed precision to avoid bf16 issues on T4
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    push_to_hub=True,
    hub_strategy="end",
    
    # Training schedule - full 3 epochs
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch = 16
    learning_rate=2e-4,
    warmup_steps=10,
    max_steps=-1,  # Use epochs instead of steps
    
    # Optimization
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    gradient_checkpointing=True,
    
    # CRITICAL: Disable ALL mixed precision (fp16 AND bf16)
    # This avoids the GradScaler bf16 issue on T4
    fp16=False,
    bf16=False,
    
    # Logging
    logging_steps=5,
    logging_first_step=True,
    report_to="none",
    
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

print("✅ Training configuration set for PRODUCTION (fp32 mode - T4 safe)")
print(f"   Training examples: {len(train_dataset)}")
print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
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
print("\n🚀 Starting PRODUCTION training...")
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
print("✅ PRODUCTION training complete!")
print(f"\n🎉 Model pushed to: https://huggingface.co/{OUTPUT_DIR}")
