# Decision Records

This project documents every major architectural and engineering decision as an Architectural Decision Record (ADR). These records serve as the project's institutional memory -- when asked "why did you do X?", the answer is traceable to a specific ADR with context, alternatives considered, rationale, and source evidence.

## Index

### Model & Training Architecture

| ADR | Title | Key Decision |
|-----|-------|-------------|
| [001](adr-001-apriori-as-oracle.md) | Apriori as Oracle | Deterministic algorithm as ground-truth source |
| [002](adr-002-qwen2.5-7b-selection.md) | Qwen2.5-7B Selection | 7B model for single-GPU fine-tuning |
| [003](adr-003-lora-vs-full-ft.md) | LoRA vs Full Fine-tuning | QLoRA for memory efficiency |
| [004](adr-004-lora-rank-32.md) | LoRA Rank 32 | r=32, alpha=64 (reduced from v2's r=64) |
| [005](adr-005-4bit-nf4.md) | 4-bit NF4 Quantization | Normal Float 4 via BitsAndBytes |
| [006](adr-006-unsloth.md) | Unsloth Framework | 2x speed, 70% VRAM reduction |

### Training Strategy

| ADR | Title | Key Decision |
|-----|-------|-------------|
| [007](adr-007-sft-before-dpo.md) | SFT Before DPO | Two-phase training order |
| [008](adr-008-dpo-not-ppo.md) | DPO Algorithm Choice | DPO over PPO/KTO/ORPO |
| [009](adr-009-real-failures-dpo.md) | Real LLM Failures for DPO | Actual model failures, not synthetic corruption |
| [010](adr-010-grpo-skipped.md) | GRPO Skipped in v3 | Focus on SFT+DPO baseline first |

### Data & Format

| ADR | Title | Key Decision |
|-----|-------|-------------|
| [011](adr-011-cot-think-tags.md) | CoT with think Tags | Qwen-native reasoning format |
| [012](adr-012-column-grouped-cot.md) | Column-Grouped CoT Format | ~40% fewer tokens, no repetition loops |
| [025](adr-025-synthetic-datasets.md) | 500 Synthetic Datasets | Full distribution control, zero licensing |

### Hyperparameters

| ADR | Title | Key Decision |
|-----|-------|-------------|
| [013](adr-013-sft-hyperparams.md) | SFT Hyperparameters | v3 changes: lr=1e-4, 3 epochs, regularization |
| [014](adr-014-dpo-hyperparams.md) | DPO Hyperparameters | lr=5e-5, beta=0.1, 1 epoch |
| [015](adr-015-seq-length-2048.md) | Sequence Length 4096 (Restored) | Initially reduced to 2048 in v3, restored to 4096 in v3.7 |
| [021](adr-021-support-and-size.md) | min_support=3, max_size=3 | Empirical sweet spot for 4-26 row datasets |

### Inference

| ADR | Title | Key Decision |
|-----|-------|-------------|
| [016](adr-016-two-phase-inference.md) | Two-Phase Inference | Separate temps for thinking and JSON |
| [017](adr-017-think-temp-0.3.md) | Thinking Temperature 0.3 | Escape deterministic attractor loops |

### Infrastructure

| ADR | Title | Key Decision |
|-----|-------|-------------|
| [018](adr-018-hash-artifact-naming.md) | Hash-Based Artifact Naming | SHA-256 for idempotent reruns |
| [019](adr-019-sqlite.md) | SQLite Persistence | Single-file database, zero setup |
| [020](adr-020-adapter-only-push.md) | Adapter-Only Model Push | LoRA adapter (~65MB) for reproducibility |

### Evaluation

| ADR | Title | Key Decision |
|-----|-------|-------------|
| [022](adr-022-validation-invariants.md) | 13 Validation Invariants | Full semantic validation |
| [023](adr-023-eval-metrics.md) | Seven Evaluation Metrics | P, R, F1, Exact Match, Hallucination, Count, Parse |
| [024](adr-024-fixed-eval-set.md) | Fixed Evaluation Set | 30 versioned datasets, zero leakage |

### Development Process

| ADR | Title | Key Decision |
|-----|-------|-------------|
| [026](adr-026-agent-workflow.md) | Agent Workflow | 9-agent AI system with persistent memory |
