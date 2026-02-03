#!/usr/bin/env python3
"""
Train Qwen model with DPO (Direct Preference Optimization).

DPO is simpler than PPO and doesn't require a separate reward model.
Based on: https://arxiv.org/abs/2305.18290

Reference implementations:
- https://github.com/huggingface/trl
- https://github.com/eric-mitchell/direct-preference-optimization
"""

import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb


@dataclass
class ScriptArguments:
    """Arguments for DPO training"""
    
    # Model arguments
    model_name: str = field(
        default="Qwen/Qwen2.5-3B-Instruct",
        metadata={"help": "Base model to fine-tune"}
    )
    
    # Data arguments
    dataset_path: str = field(
        default="data/hf_rlhf_dataset_v1",
        metadata={"help": "Path to HuggingFace dataset"}
    )
    
    # LoRA arguments
    use_lora: bool = field(
        default=True,
        metadata={"help": "Use LoRA for parameter-efficient training"}
    )
    lora_r: int = field(default=64, metadata={"help": "LoRA attention dimension"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    
    # Training arguments
    output_dir: str = field(
        default="./dpo_checkpoints",
        metadata={"help": "Output directory for model checkpoints"}
    )
    num_train_epochs: int = field(default=3, metadata={"help": "Number of epochs"})
    per_device_train_batch_size: int = field(default=1, metadata={"help": "Train batch size"})
    per_device_eval_batch_size: int = field(default=1, metadata={"help": "Eval batch size"})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "Gradient accumulation"})
    learning_rate: float = field(default=5e-5, metadata={"help": "Learning rate"})
    max_length: int = field(default=2048, metadata={"help": "Max sequence length"})
    max_prompt_length: int = field(default=1024, metadata={"help": "Max prompt length"})
    
    # DPO-specific arguments
    beta: float = field(
        default=0.1,
        metadata={"help": "DPO temperature parameter (controls strength of preference)"}
    )
    
    # Quantization
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Use 4-bit quantization"}
    )
    
    # Logging
    use_wandb: bool = field(default=False, metadata={"help": "Log to W&B"})
    wandb_project: str = field(
        default="itemset-dpo",
        metadata={"help": "W&B project name"}
    )
    
    # Evaluation
    eval_steps: int = field(default=50, metadata={"help": "Evaluation frequency"})
    save_steps: int = field(default=100, metadata={"help": "Save frequency"})


def format_example(example, tokenizer):
    """
    Format DPO example for training.
    
    Input example format (from create_rlhf_hf_dataset.py):
    {
        "prompt": [{"role": "system", ...}, {"role": "user", ...}],
        "chosen": [{"role": "assistant", ...}],
        "rejected": [{"role": "assistant", ...}]
    }
    """
    # Apply chat template to prompt
    prompt_text = tokenizer.apply_chat_template(
        example["prompt"], 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Get chosen and rejected responses
    chosen_text = example["chosen"][0]["content"]
    rejected_text = example["rejected"][0]["content"]
    
    return {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


def main():
    # Parse arguments
    parser = HfArgumentParser((ScriptArguments,))
    script_args = parser.parse_args_into_dataclasses()[0]
    
    print("=" * 60)
    print("🚀 Starting DPO Training")
    print("=" * 60)
    print(f"Model: {script_args.model_name}")
    print(f"Dataset: {script_args.dataset_path}")
    print(f"Output: {script_args.output_dir}")
    print(f"Use LoRA: {script_args.use_lora}")
    print(f"Use 4-bit: {script_args.use_4bit}")
    print(f"DPO Beta: {script_args.beta}")
    print("=" * 60)
    
    # Initialize W&B
    if script_args.use_wandb:
        wandb.init(
            project=script_args.wandb_project,
            name=f"dpo-{Path(script_args.model_name).name}",
            config=vars(script_args),
        )
    
    # Load tokenizer
    print("\n📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name,
        trust_remote_code=True,
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"\n🤖 Loading model: {script_args.model_name}")
    
    if script_args.use_4bit:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    
    model.config.use_cache = False
    
    # Apply LoRA
    if script_args.use_lora:
        print("\n🔧 Applying LoRA...")
        
        if script_args.use_4bit:
            model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Load dataset
    print(f"\n📦 Loading dataset from {script_args.dataset_path}")
    dataset = load_from_disk(script_args.dataset_path)
    
    print(f"   Train examples: {len(dataset['train'])}")
    print(f"   Val examples: {len(dataset['validation'])}")
    
    # Format dataset
    print("\n🔄 Formatting dataset...")
    
    def format_dataset(examples):
        formatted = []
        for i in range(len(examples["prompt"])):
            example = {
                "prompt": examples["prompt"][i],
                "chosen": examples["chosen"][i],
                "rejected": examples["rejected"][i],
            }
            formatted.append(format_example(example, tokenizer))
        
        return {
            "prompt": [ex["prompt"] for ex in formatted],
            "chosen": [ex["chosen"] for ex in formatted],
            "rejected": [ex["rejected"] for ex in formatted],
        }
    
    train_dataset = dataset["train"].map(
        format_dataset,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    eval_dataset = dataset["validation"].map(
        format_dataset,
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )
    
    print(f"   Formatted train: {len(train_dataset)} examples")
    print(f"   Formatted val: {len(eval_dataset)} examples")
    
    # Training arguments
    training_args = DPOConfig(
        output_dir=script_args.output_dir,
        num_train_epochs=script_args.num_train_epochs,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        max_length=script_args.max_length,
        max_prompt_length=script_args.max_prompt_length,
        beta=script_args.beta,
        
        # Optimization
        optim="paged_adamw_8bit" if script_args.use_4bit else "adamw_torch",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        
        # Logging & evaluation
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=script_args.eval_steps,
        save_strategy="steps",
        save_steps=script_args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        
        # W&B
        report_to="wandb" if script_args.use_wandb else "none",
        
        # Misc
        warmup_steps=50,
        remove_unused_columns=False,
    )
    
    # Create DPO trainer
    print("\n🎯 Creating DPO Trainer...")
    
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\n🚂 Starting training...")
    print("=" * 60)
    
    dpo_trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    output_dir = Path(script_args.output_dir)
    final_model_dir = output_dir / "final_model"
    
    dpo_trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    print(f"\n✅ Training complete!")
    print(f"   Final model: {final_model_dir}")
    
    if script_args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
