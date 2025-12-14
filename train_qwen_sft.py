"""
Supervised Fine-Tuning (SFT) script for Qwen2.5-0.5B
Fine-tune on frequent itemset extraction task using LoRA/QLoRA
"""

import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import argparse


def format_chat_template(example, tokenizer):
    """Format messages using the model's chat template"""
    # Extract messages from the example
    messages = example["messages"]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-0.5B on itemset extraction")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model name")
    parser.add_argument("--dataset-path", default="hf_dataset", help="Path to HF dataset")
    parser.add_argument("--output-dir", default="qwen_itemset_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    print("🚀 Starting Qwen2.5-0.5B Fine-Tuning for Itemset Extraction")
    print(f"   Model: {args.model_name}")
    print(f"   Dataset: {args.dataset_path}")
    print(f"   4-bit: {args.use_4bit}")
    print(f"   LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    
    # Load dataset
    print("\n📦 Loading dataset...")
    dataset = load_from_disk(args.dataset_path)
    print(f"   Train: {len(dataset['train'])} examples")
    print(f"   Validation: {len(dataset['validation'])} examples")
    
    # Configure quantization (optional)
    if args.use_4bit:
        print("\n⚙️  Configuring 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("\n🤖 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config if args.use_4bit else None,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not args.use_4bit else None,
    )
    
    # Prepare model for training
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    print("\n🔧 Configuring LoRA...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Training arguments
    print("\n⚙️  Configuring training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=False,
        bf16=True,
        group_by_length=True,
        report_to=["tensorboard"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Format dataset
    print("\n📋 Formatting dataset with chat template...")
    train_dataset = dataset["train"].map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset["train"].column_names,
    )
    eval_dataset = dataset["validation"].map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset["validation"].column_names,
    )
    
    # Create trainer
    print("\n🏋️  Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )
    
    # Train
    print("\n🚀 Starting training...")
    print("=" * 60)
    trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    
    print("\n✅ Training completed!")
    print(f"   Model saved to: {args.output_dir}/final")
    print(f"\nTo use the model:")
    print(f"   from peft import AutoPeftModelForCausalLM")
    print(f"   model = AutoPeftModelForCausalLM.from_pretrained('{args.output_dir}/final')")


if __name__ == "__main__":
    main()
