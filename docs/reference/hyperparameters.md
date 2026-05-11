# Hyperparameter Reference

Every numeric parameter used in this project, sourced to the exact file and line.

## Pipeline Parameters

| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| min_support | 3 | `pipeline.py:595` | Apriori minimum support count |
| max_size | 3 | `pipeline.py:596` | Maximum itemset size (singles, pairs, triples) |
| chunk_size | 50 | `pipeline.py:600` | Rows per LLM API call |
| llm_timeout | 180s | `pipeline.py:486` | Per-request timeout |
| llm_retries | 2 | `pipeline.py:486` | Retry count on timeout |
| llm_temperature | 0.0 | `pipeline.py:483` | Extraction temperature (not used for reasoning models) |

## Training Data Generation

| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| val_ratio | 0.10 | `generate_cot_sft_data.py` / `export_real_dpo_data.py` | 10% validation split |
| seed | 42 | `generate_cot_sft_data.py` / `export_real_dpo_data.py` | All data splits + LoRA init |
| max_sft_tokens | 3500 | `generate_cot_sft_data.py:69` | Per-example token budget (chars/4 estimate) |
| max_cot_items | 40 | `generate_cot_sft_data.py:73` | Items shown in CoT before abbreviation |
| max_dpo_rejected | 3 | `export_real_dpo_data.py:88` | Max rejected outputs per dataset |

## LoRA Configuration

| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| r (rank) | 32 | `training_3phase_v3.ipynb` Cell 2 | Was 64 in v2 |
| alpha | 64 | `training_3phase_v3.ipynb` Cell 2 | alpha/r = 2.0 ratio |
| dropout | 0.05 | `training_3phase_v3.ipynb` Cell 2 | Added in v3 (was 0) |
| target_modules | q, k, v, o, gate, up, down | `training_3phase_v3.ipynb` Cell 2 | All attention + MLP |
| bias | "none" | `training_3phase_v3.ipynb` Cell 2 | No bias training |
| random_state | 42 | `training_3phase_v3.ipynb` Cell 7 | LoRA weight initialization |

## SFT Training

| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| learning_rate | 1e-4 | `training_3phase_v3.ipynb` Cell 2 | Was 2e-4 in v2 |
| per_device_batch_size | 2 | `training_3phase_v3.ipynb` Cell 2 | |
| gradient_accumulation_steps | 4 | `training_3phase_v3.ipynb` Cell 2 | Effective batch = 8 |
| num_train_epochs | 3 | `training_3phase_v3.ipynb` Cell 2 | Was 2 in v2 |
| warmup_ratio | 0.10 | `training_3phase_v3.ipynb` Cell 2 | Was 0.05 in v2 |
| weight_decay | 0.01 | `training_3phase_v3.ipynb` Cell 2 | Added in v3 |
| max_seq_length | 4096 | `training_3phase_v3.ipynb` Cell 2 | Initially reduced to 2048 in v3, restored to 4096 in v3.7 to avoid truncating SFT targets |
| optimizer | AdamW | TRL default | |
| lr_scheduler | Linear warmup + cosine | TRL default | |
| gradient_checkpointing | "unsloth" | `training_3phase_v3.ipynb` Cell 2 | Memory optimization |

## DPO Training

| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| learning_rate | 5e-5 | `training_3phase_v3.ipynb` Cell 2 | Lower than SFT |
| beta | 0.1 | `training_3phase_v3.ipynb` Cell 2 | KL penalty weight |
| per_device_batch_size | 1 | `training_3phase_v3.ipynb` Cell 2 | Chosen+rejected doubles memory |
| gradient_accumulation_steps | 4 | `training_3phase_v3.ipynb` Cell 2 | Effective batch = 4 |
| num_train_epochs | 1 | `training_3phase_v3.ipynb` Cell 2 | Was 2 in v2 |
| loss_type | "sigmoid" | TRL DPOTrainer default | Standard DPO loss |

## Inference

| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| think_temperature | 0.3 | `inference_utils.py:177` | Phase 1: reasoning |
| json_temperature | 0.05 | `inference_utils.py:179` | Phase 2: structured output |
| max_generation_tokens | 6000 | `inference_utils.py:115` | Hard cap per generation |

## Quantization

| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| load_in_4bit | True | `training_3phase_v3.ipynb` Cell 2 | BitsAndBytes 4-bit |
| bnb_4bit_quant_type | "nf4" | `training_3phase_v3.ipynb` Cell 2 | Normal Float 4 |
| bnb_4bit_compute_dtype | bfloat16 | `training_3phase_v3.ipynb` Cell 2 | Compute precision |
