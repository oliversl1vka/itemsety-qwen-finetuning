# Documentation Execution Plan

> **Purpose:** One-shot reference for writing the MkDocs documentation site.  
> **Principle:** Docs are written LAST, on a clean repo, in a single pass.  
> See `CLEANUP_CHECKLIST.md` for everything to complete before starting.

---

## 1. Setup

### Install

```bash
pip install mkdocs-material mkdocstrings[python] pymdown-extensions
```

### Preview locally

```bash
mkdocs serve
# → http://127.0.0.1:8000
```

### Deploy to GitHub Pages (one command)

```bash
mkdocs gh-deploy
```

---

## 2. `mkdocs.yml` — full configuration

Place at repo root. Copy-paste ready (fill in `YOUR_GITHUB_USERNAME`).

```yaml
site_name: Itemset Extraction via Fine-tuned LLM
site_description: >
  Bachelor's thesis — fine-tuning Qwen2.5-7B to extract frequent itemsets
  using Apriori as a deterministic ground-truth oracle. No human annotation required.
site_url: https://YOUR_GITHUB_USERNAME.github.io/itemsety-qwen-finetuning/
repo_url: https://github.com/YOUR_GITHUB_USERNAME/itemsety-qwen-finetuning
repo_name: itemsety-qwen-finetuning
edit_uri: ""

theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  font:
    text: Inter
    code: JetBrains Mono
  features:
    - navigation.tabs
    - navigation.tabs.sticky
    - navigation.sections
    - navigation.expand
    - navigation.path
    - navigation.top
    - navigation.footer
    - search.suggest
    - search.highlight
    - content.code.annotate
    - content.code.copy
    - content.tabs.link
    - toc.follow

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [.]
          options:
            show_source: true
            show_root_heading: true
            show_symbol_type_heading: true
            docstring_style: google
            merge_init_into_class: true

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.details
  - pymdownx.emoji:
      emoji_index: !!python/name:material.extensions.emoji.twemoji
      emoji_generator: !!python/name:material.extensions.emoji.to_svg
  - pymdownx.arithmatex:
      generic: true
  - admonition
  - tables
  - attr_list
  - md_in_html
  - footnotes
  - toc:
      permalink: true

extra_javascript:
  - javascripts/mathjax.js
  - https://polyfill.io/v3/polyfill.min.js?features=es6
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js

extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/YOUR_GITHUB_USERNAME/itemsety-qwen-finetuning
    - icon: simple/huggingface
      link: https://huggingface.co/OliverSlivka

nav:
  - Home: index.md
  - Quick Start: quickstart.md
  - Methodology:
    - Overview: methodology/overview.md
    - Data Generation: methodology/data-generation.md
    - SFT Training: methodology/sft-training.md
    - DPO Training: methodology/dpo-training.md
    - Evaluation: methodology/evaluation.md
  - Decision Records:
    - Index: decisions/index.md
    - "ADR-001: Apriori as Oracle": decisions/adr-001-apriori-as-oracle.md
    - "ADR-002: Qwen2.5-7B Selection": decisions/adr-002-qwen2.5-7b-selection.md
    - "ADR-003: LoRA vs Full Fine-tuning": decisions/adr-003-lora-vs-full-ft.md
    - "ADR-004: LoRA Rank 32": decisions/adr-004-lora-rank-32.md
    - "ADR-005: 4-bit NF4 Quantization": decisions/adr-005-4bit-nf4.md
    - "ADR-006: Unsloth Framework": decisions/adr-006-unsloth.md
    - "ADR-007: SFT Before DPO": decisions/adr-007-sft-before-dpo.md
    - "ADR-008: DPO Algorithm Choice": decisions/adr-008-dpo-not-ppo.md
    - "ADR-009: Real LLM Failures for DPO": decisions/adr-009-real-failures-dpo.md
    - "ADR-010: GRPO Skipped in v3": decisions/adr-010-grpo-skipped.md
    - "ADR-011: CoT with think Tags": decisions/adr-011-cot-think-tags.md
    - "ADR-012: Column-Grouped CoT Format": decisions/adr-012-column-grouped-cot.md
    - "ADR-013: SFT Hyperparameters": decisions/adr-013-sft-hyperparams.md
    - "ADR-014: DPO Hyperparameters": decisions/adr-014-dpo-hyperparams.md
    - "ADR-015: Sequence Length 2048": decisions/adr-015-seq-length-2048.md
    - "ADR-016: Two-Phase Inference": decisions/adr-016-two-phase-inference.md
    - "ADR-017: Thinking Temperature 0.3": decisions/adr-017-think-temp-0.3.md
    - "ADR-018: Hash-Based Artifact Naming": decisions/adr-018-hash-artifact-naming.md
    - "ADR-019: SQLite Persistence": decisions/adr-019-sqlite.md
    - "ADR-020: Adapter-Only Model Push": decisions/adr-020-adapter-only-push.md
    - "ADR-021: min_support=3, max_size=3": decisions/adr-021-support-and-size.md
    - "ADR-022: 13 Validation Invariants": decisions/adr-022-validation-invariants.md
    - "ADR-023: Seven Evaluation Metrics": decisions/adr-023-eval-metrics.md
    - "ADR-024: Fixed Evaluation Set": decisions/adr-024-fixed-eval-set.md
    - "ADR-025: 500 Synthetic Datasets": decisions/adr-025-synthetic-datasets.md
    - "ADR-026: Agent Workflow": decisions/adr-026-agent-workflow.md
  - Reference:
    - Hyperparameters: reference/hyperparameters.md
    - Database Schema: reference/database-schema.md
    - Prompt Templates: reference/prompt-templates.md
    - Code Reference: reference/code-reference.md
  - AI Workflow:
    - Overview: ai-workflow/overview.md
    - Agent System: ai-workflow/agent-system.md
    - Experiment Journal: ai-workflow/experiment-journal.md
  - Glossary: glossary.md
```

---

## 3. GitHub Actions — Automatic Pages Deploy

Create `.github/workflows/docs.yml`:

```yaml
name: Deploy Documentation

on:
  push:
    branches: [main]
    paths:
      - 'docs/**'
      - 'mkdocs.yml'
      - '.github/workflows/docs.yml'
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install MkDocs
        run: pip install mkdocs-material mkdocstrings[python] pymdown-extensions
      - name: Build site
        run: mkdocs build --strict
      - uses: actions/upload-pages-artifact@v3
        with:
          path: site/
      - id: deployment
        uses: actions/deploy-pages@v4
```

Enable in GitHub: Settings → Pages → Source → **GitHub Actions**.

Also create `docs/javascripts/mathjax.js`:
```javascript
window.MathJax = {
  tex: { inlineMath: [["\\(", "\\)"]], displayMath: [["\\[", "\\]"]] },
  svg: { fontCache: "global" }
};
```

---

## 4. Page-by-Page Content Specification

---

### `docs/index.md` — Landing Page

**Content:**

1. **Hero** — one paragraph: what this proves + key result (fill `F1_SCORE` placeholder)
   > *"This repository presents an end-to-end ML research pipeline that fine-tunes Qwen2.5-7B to extract frequent itemsets from tabular data. Using Apriori as a deterministic, annotation-free ground-truth oracle, the system achieves F1=**F1_SCORE** on itemset extraction — outperforming GPT-4.1-mini baseline. Training used SFT with chain-of-thought reasoning followed by DPO on real LLM failure examples."*

2. **Links row** — badges/buttons:
   - Model on HuggingFace: `OliverSlivka/qwen2.5-7b-itemset-extractor`
   - Dataset on HuggingFace: `OliverSlivka/itemset-extraction-v3`
   - Thesis PDF (if publicly available)
   - Demo (Gradio Space if live)

3. **Results table** (fill from your eval run):

   | Model | Precision | Recall | F1 | Exact Match |
   |---|---|---|---|---|
   | Base Qwen2.5-7B | ? | ? | ? | ? |
   | + SFT (phase 1) | ? | ? | ? | ? |
   | + DPO (phase 2) | ? | ? | ? | ? |
   | GPT-4.1-mini baseline | ? | ? | ? | ? |

4. **Architecture diagram** (Mermaid):
   ```
   CSV Datasets → Apriori Oracle → LLM Pipeline → runs.db
   runs.db → SFT Data (348 ex) → Phase 1 SFT
   runs.db → DPO Data (606 pairs) → Phase 2 DPO
   Fine-tuned Model → Evaluation → Metrics
   ```

5. **Key design decisions** — 3 bullets:
   - Apriori as a self-validating oracle eliminates human annotation
   - Real LLM failure examples (not synthetic) for DPO
   - Two-phase inference separates reasoning from structured JSON output

6. **Citation block** (BibTeX)

7. **License + Author**

---

### `docs/quickstart.md` — Quick Start

**Content:**

1. **Prerequisites**: Python 3.10+, Git, API keys (OpenAI or OpenRouter), optional GPU
2. **Clone & install**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/itemsety-qwen-finetuning
   cd itemsety-qwen-finetuning
   pip install -r requirements.txt
   ```
3. **Configure credentials**:
   ```bash
   cp openai.env.template openai.env
   cp openrouter.env.template openrouter.env
   cp hf.env.template hf.env
   # Edit each file and fill in your keys
   ```
4. **Run the pipeline on one dataset**:
   ```bash
   python pipeline.py --data data/datasets_v2/ds_0001_7x12.csv \
     --min-support 3 --max-size 3 --llm-full --llm-model gpt-4.1-mini
   ```
5. **Expected output**: JSON artifact + SQLite row in `runs.db`
6. **Run evaluation against fine-tuned model**:
   ```bash
   python src/evaluation/eval_finetuned_model.py \
     --model-path OliverSlivka/qwen2.5-7b-itemset-extractor
   ```
7. **Reproduce training** — note: requires A100/H100 GPU (24GB+ VRAM), link to notebook
8. **Data generation** (if regenerating from scratch):
   ```bash
   python pipeline.py --data-dir data/datasets_v2 \
     --min-support 3 --max-size 3 --llm-full --llm-model gpt-4.1-mini
   python src/training/generate_cot_sft_data.py --db runs.db --output data/sft_cot_v3.json
   python src/training/export_real_dpo_data.py --db runs.db --output data/dpo_real_v2.json
   python src/training/build_hf_dataset_v2.py \
     --sft data/sft_cot_v3.json --dpo data/dpo_real_v2.json --output data/hf_dataset_v3
   ```

---

### `docs/methodology/overview.md` — Methodology Overview

**Content:**

1. **Research problem**: Frequent itemset extraction requires scanning all rows for co-occurring items — a structured reasoning task. This work teaches an LLM to do this without human-labeled training data.

2. **Full pipeline Mermaid diagram** (large, detailed):
   ```
   flowchart TD
     A[Generate 500 synthetic CSVs] --> B[Apriori ground truth]
     B --> C[LLM extraction via pipeline.py]
     C --> D[(runs.db - 1600+ runs)]
     D --> E[generate_cot_sft_data.py]
     D --> F[export_real_dpo_data.py]
     E --> G[348 SFT examples with CoT]
     F --> H[606 DPO preference pairs]
     G --> I[Phase 1: SFT-CoT]
     H --> J[Phase 2: DPO]
     I --> J
     J --> K[Fine-tuned Qwen2.5-7B]
     K --> L[Evaluation vs Apriori]
   ```

3. **Why this approach** — three key insights:
   - Apriori is deterministic → free ground truth → no annotation bottleneck
   - CoT training teaches the reasoning process, not just the answer
   - DPO on real failures corrects the specific failure mode (hallucinated evidence)

4. **Training phases overview table**:

   | Phase | Method | Data | Key Hyperparams | Goal |
   |---|---|---|---|---|
   | 1 | SFT-CoT | 348 examples | lr=1e-4, 3 epochs | Learn format + reasoning |
   | 2 | DPO | 606 pairs | lr=5e-5, β=0.1, 1 epoch | Reject hallucinated evidence |
   | 3 | GRPO | — | Skipped in v3 | (future work) |

5. Links to sub-pages for each methodology section.

---

### `docs/methodology/data-generation.md` — Data Generation

**Content:**

1. **Dataset naming convention**: `ds_<NNNN>_<rows>x<cols>.csv` — 500 files, rows 5–25, cols 3–15
   - Source: `src/data_generation/generate_datasets_v2.py`

2. **Dataset structure**: each CSV has column headers as item categories, cells as item values; each row is a transaction

3. **Why synthetic** — see ADR-025; key point: full distribution control, zero licensing friction, reproducible

4. **Apriori as oracle**:
   - Algorithm: standard Apriori with sorted-item canonical form
   - Parameters: `min_support=3`, `max_size=3` (see ADR-021)
   - Source: `pipeline.py:401-466`
   - Deterministic: same CSV always → same output → no label noise

5. **Pipeline execution** — describe the 5-step flow with diagram:
   ```
   CSV load → format detection (wide/long/single-col) → Apriori → LLM extract → validate → SQLite
   ```
   - Auto-detects CSV format: `pipeline.py:338-399`
   - Runs all 500 datasets: `--data-dir data/datasets_v2`

6. **runs.db** — the central artifact:
   - 1600+ rows (multiple LLM models × 500 datasets)
   - Feeds both SFT and DPO generation scripts
   - 26 columns: timestamp, dataset_id, dataset_hash, min_support, max_size, llm_model, validation_passed, apriori/llm counts, etc.
   - Not committed to git (gitignored) — regenerable from pipeline

7. **SFT data generation** (`src/training/generate_cot_sft_data.py`):
   - Queries runs.db for one valid Apriori run per unique dataset
   - Generates CoT reasoning in v3 column-grouped format (see ADR-012)
   - Filters: max 3500 tokens per example; skips empty Apriori outputs
   - Output: 348 examples (314 train / 34 val, seed=42, 10% split)

8. **DPO data generation** (`src/training/export_real_dpo_data.py`):
   - Queries runs.db: `validation_passed=0 AND llm_itemset_count > 0`
   - Sources rejected outputs from 4 models: GPT-4.1-mini, GPT-4.1-nano, GPT-4o, o4-mini
   - Max 3 rejected outputs per dataset (prevents over-representation)
   - Output: 606 pairs (546 train / 60 val, seed=42, 10% split)

---

### `docs/methodology/sft-training.md` — SFT Training (Phase 1)

**Content:**

1. **What is SFT**: brief explanation; goal is to teach format + reasoning

2. **Training data format** — show the exact message structure:
   ```json
   {
     "messages": [
       {"role": "system", "content": "You are a frequent itemset extractor..."},
       {"role": "user", "content": "Find all frequent itemsets with min_support=3..."},
       {"role": "assistant", "content": "<think>\n[CoT reasoning]\n</think>\n[{\"itemset\": [...]}]"}
     ]
   }
   ```

3. **CoT Format v3 (column-grouped)** — show an example:
   - Header: "Dataset: N rows, M cols. min_support=K."
   - SINGLES SCAN (column-grouped): `col:val=count(R1,R2,R3)✓`
   - FREQUENT SINGLES: `[n] items: item1, item2, ...`
   - PAIRS SCAN: `(item1,item2)=count(R1,R2)✓`
   - TRIPLES SCAN
   - RESULT SUMMARY: `N singles + M pairs + K triples = TOTAL itemsets`
   - Source: `src/training/training_utils.py:60-191`

4. **System prompt** (verbatim, ~150 tokens):
   - Source: `src/training/training_utils.py:22-34`

5. **Complete hyperparameter table** (all sourced to notebook Cell 2):

   | Parameter | Value | Notes |
   |---|---|---|
   | Base model | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | Pre-quantized Unsloth format |
   | Quantization | 4-bit NF4 | bfloat16 compute via BitsAndBytes |
   | LoRA rank (r) | 32 | Reduced from 64 in v2 (see ADR-004) |
   | LoRA alpha (α) | 64 | α/r = 2.0 ratio |
   | LoRA dropout | 0.05 | Added in v3 (was 0 in v2) |
   | LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All attention + MLP |
   | Learning rate | 1e-4 | Reduced from 2e-4 in v2 |
   | Batch size | 2 (per device) | |
   | Gradient accumulation | 4 | Effective batch = 8 |
   | Epochs | 3 | Increased from 2 in v2 |
   | Warmup ratio | 0.10 | Increased from 0.05 in v2 |
   | Weight decay | 0.01 | Added in v3 |
   | Max sequence length | 2048 | Reduced from 4096 in v2 |
   | Optimizer | AdamW | TRL default |
   | Scheduler | Linear warmup + cosine | TRL default |
   | Packing | Yes | TRL SFTTrainer default |
   | Gradient checkpointing | "unsloth" | Memory optimization |
   | Random seed | 42 | Data split + LoRA init |

6. **Framework**: Unsloth `FastLanguageModel` + TRL `SFTTrainer` + `train_on_responses_only`

7. **Save strategy**: adapter-only push (see ADR-020) — `~65MB` vs `~14GB` merged

---

### `docs/methodology/dpo-training.md` — DPO Training (Phase 2)

**Content:**

1. **What is DPO**: brief conceptual explanation; preferred vs rejected response pairs; loss formula (optional with LaTeX): `L_DPO = -E[log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))]`

2. **Why DPO over PPO** — see ADR-008; key point: no reward model needed, stable training, our data is naturally paired

3. **Data composition**:
   - 606 preference pairs (546 train / 60 val)
   - Chosen: Apriori correct output (with v3 CoT reasoning)
   - Rejected: Real LLM failure from one of 4 models
   - Source: `src/training/export_real_dpo_data.py`

4. **Error distribution pie chart / table**:
   - 99.5%: `item_missing_in_row` — item in predicted itemset doesn't appear in the cited row
   - 0.5%: other errors (count mismatch, JSON malformation)
   - Source: `export_real_dpo_data.py:11-14`

5. **Why real failures beat synthetic** — ADR-009 reference; core argument: synthetic random corruption doesn't match actual LLM behavior patterns

6. **DPO hyperparameter table**:

   | Parameter | Value | Notes |
   |---|---|---|
   | Learning rate | 5e-5 | Lower than SFT (DPO more sensitive) |
   | DPO beta (β) | 0.1 | KL penalty; higher = more conservative |
   | Batch size | 1 (per device) | chosen+rejected doubles effective memory |
   | Gradient accumulation | 4 | Effective batch = 4 |
   | Epochs | 1 | Reduced from 2 (DPO overfits quickly) |
   | Loss type | Standard DPO | TRL DPOTrainer default |
   | Reference model | SFT checkpoint | Frozen SFT model as π_ref |
   | Random seed | 42 | |

7. **Framework**: TRL `DPOTrainer`, requires processing_class parameter (not tokenizer — fixed in commit e195983)

---

### `docs/methodology/evaluation.md` — Evaluation

**Content:**

1. **Evaluation philosophy**: same Apriori algorithm used as oracle during eval — consistent ground truth, no human judgment in the loop

2. **Fixed evaluation set** (see ADR-024):
   - 20 datasets from `data/eval_datasets_v2/`
   - Versioned, never changed between model comparisons
   - Explicitly excluded from training data by filename matching

3. **Seven metrics** with formulas and intuition:

   | Metric | Formula | What it measures |
   |---|---|---|
   | Precision | TP / (TP + FP) | Are predicted itemsets correct? |
   | Recall | TP / (TP + FN) | Are all true itemsets found? |
   | F1 | 2·P·R / (P+R) | Harmonic mean — primary metric |
   | Exact Match | F1 == 1.0 | Full correctness (binary per dataset) |
   | Count Accuracy | correct_counts / matched | Support count within ±1 tolerance |
   | Hallucination Rate | FP items not in CSV / all predicted items | Primary failure mode tracking |
   | JSON Parse Rate | parseable outputs / total outputs | Format adherence |

   Source: `src/evaluation/eval_finetuned_model.py:417-500`

4. **Canonical form** for comparison: `frozenset(str(x).strip().lower() for x in itemset)` — source: `eval_finetuned_model.py:412-414`

5. **Inference configuration during eval**:
   - Two-phase (see ADR-016): think at temp=0.3, JSON at temp=0.05
   - ThinkStoppingCriteria halts at `</think>` token sequence
   - RepetitionDetector catches infinite loops
   - Max 6000 tokens hard cap
   - Source: `src/evaluation/inference_utils.py:159-245`

6. **Results table** (FILL IN from your eval run):

   | Model | P | R | F1 | Exact Match | Hallucination Rate | JSON Parse |
   |---|---|---|---|---|---|---|
   | Base Qwen2.5-7B | ? | ? | ? | ? | ? | ? |
   | + SFT v3 | ? | ? | ? | ? | ? | ? |
   | + SFT+DPO v3 | ? | ? | ? | ? | ? | ? |
   | GPT-4.1-mini | ? | ? | ? | ? | ? | ? |

7. **Size-wise breakdown** (per itemset size 1, 2, 3) — fill from eval output

---

## 5. Decision Records — Full Content Specifications

All ADRs use this template:

```markdown
---
title: ADR-XXX — [Title]
status: Accepted
date: YYYY-MM-DD
---

## Context
[What problem? What was the situation before this decision?]

## Options Considered
| Option | Pros | Cons |
|--------|------|------|

## Decision
**[Chosen option]** — one sentence why.

## Rationale
[Detailed reasoning paragraphs]

## Trade-offs
[What was accepted/given up]

## Source Evidence
- `file.py:L-L` — [what it shows]

## Consequences
[Impact on the system / downstream decisions this enabled]
```

---

### ADR-001: Apriori as Oracle
- **Date**: 2026-02-XX (when pipeline.py was first written)
- **Context**: Need ground truth for training without human annotation. Labeling 500 datasets × all itemsets manually is infeasible.
- **Options**: Human labeling, Apriori, FP-Growth, LLM-as-judge
- **Decision**: Apriori — deterministic, zero annotation cost, mathematically exact
- **Rationale**: Fully deterministic → reproducible → no label noise → enables a self-supervised pipeline. Same algorithm used for both training and evaluation ensuring consistent measurement. LLM-as-judge introduces circular reasoning.
- **Trade-offs**: Capped at max_size=3; no fuzzy matching; limited to exact Apriori semantics
- **Source**: `pipeline.py:401-466` (apriori_frequent_itemsets)

### ADR-002: Qwen2.5-7B Selection
- **Date**: 2026-02-XX
- **Context**: Choose base model for fine-tuning under GPU memory constraint (single A100 40GB or T4 16GB)
- **Options**: Qwen2.5-3B (too small), Qwen2.5-7B (chosen), Qwen2.5-14B (too large), Llama-3-8B, Mistral-7B
- **Decision**: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- **Rationale**: 7B fits in 4-bit on T4/A100. Instruct variant follows structured output prompts natively. Qwen2.5 has native `<think>` tag support matching our CoT format. Stronger reasoning benchmarks vs Llama-3 at same size. Unsloth has first-class Qwen2.5 support (2× speed advantage).
- **Trade-offs**: 14B would be stronger but requires 2× VRAM; 7B is the efficiency frontier for single-GPU deployment
- **Source**: `notebooks/training_3phase_7b.ipynb` Cell 2 (model_id config)

### ADR-003: LoRA vs Full Fine-tuning
- **Context**: How to adapt the model weights
- **Options**: Full fine-tuning (~28B optimizer params), LoRA, QLoRA
- **Decision**: QLoRA (4-bit quantized base + LoRA adapters)
- **Rationale**: Full FT requires >100GB VRAM for 7B model + optimizer states. LoRA adapts a small fraction of parameters (r=32 means ~0.05% of parameters) with minimal quality degradation. Adapter storage is ~65MB vs ~14GB merged. QLoRA enables training on consumer GPUs.
- **Trade-offs**: Quality ceiling vs full FT; requires base model for inference
- **Source**: `notebooks/training_3phase_7b.ipynb` Cell 7 (`FastLanguageModel.get_peft_model`)

### ADR-004: LoRA Rank 32
- **Context**: r and alpha values for LoRA configuration
- **Options**: r=8 (too low expressivity), r=16, r=32 (chosen), r=64 (v2, overfitting), r=128
- **Decision**: r=32, α=64 (ratio 2.0)
- **Rationale**: v2 used r=64 and showed signs of overfitting on only 348 SFT examples. r=32 reduces overfit risk while retaining sufficient expressivity for the structured JSON task. α=64 maintains the standard 2× scaling factor. Council analysis (2026-03-09) confirmed this change.
- **Trade-offs**: Slightly less capacity for complex pattern generalization
- **Source**: `notebooks/training_3phase_2026-03-09_v3.ipynb` Cell 2 (v3 change comment)

### ADR-005: 4-bit NF4 Quantization
- **Context**: Memory reduction strategy for 7B model
- **Options**: bf16 full precision (~28GB), 8-bit (~14GB), 4-bit NF4 (~7GB), 4-bit FP4
- **Decision**: 4-bit NF4 (Normal Float 4) via BitsAndBytes
- **Rationale**: NF4 is information-theoretically optimal for normally-distributed weights — neural network weights follow approximately normal distributions, so NF4 quantization levels are placed at optimal points. Provides 70% VRAM reduction vs bf16. Unsloth + bnb 4-bit enables 2× training speedup. Enables training on T4 (16GB).
- **Trade-offs**: Requires adapter-only push (can't merge NF4 back cleanly — see ADR-020)
- **Source**: `notebooks/training_3phase_7b.ipynb` Cell 2 (load_in_4bit=True, bnb_4bit_quant_type="nf4")

### ADR-006: Unsloth Framework
- **Context**: Choose the training framework layer above HuggingFace Transformers
- **Options**: Raw Transformers + PEFT, Axolotl, LLaMA-Factory, Unsloth
- **Decision**: Unsloth
- **Rationale**: 2× faster training vs raw PEFT via custom CUDA kernels for attention. 70% less VRAM (verified). `train_on_responses_only` helper is critical for SFT — it masks loss on user/system turns so only assistant output is trained. Native Qwen2.5 + BitsAndBytes 4-bit support. Active development. `FastLanguageModel.for_inference` provides optimized inference path.
- **Trade-offs**: Black-box CUDA kernels harder to debug; less flexible than raw Transformers
- **Source**: `requirements.txt` (unsloth), `notebooks/training_3phase_7b.ipynb` Cell 1 (`from unsloth import FastLanguageModel`)

### ADR-007: SFT Before DPO
- **Context**: Should we do DPO on the base model or SFT-first?
- **Options**: DPO-only from base, SFT then DPO (chosen), joint training, RLHF (PPO)
- **Decision**: SFT (phase 1) → DPO (phase 2)
- **Rationale**: The base model has no knowledge of our output format (`<think>` + JSON array of itemsets). DPO on the base model diverges catastrophically because the reference policy has no concept of the task. SFT first establishes: (1) the JSON output schema, (2) the CoT reasoning pattern, (3) the system prompt response style. DPO then refines preference on an already-capable model — a much smaller distribution shift.
- **Trade-offs**: 2-phase training doubles wall time; SFT quality ceiling constrains DPO ceiling
- **Source**: `obsidian-brain/Experiments/2026-03-09 Council v3 Plan.md`; notebook structure (SFT cell before DPO cell)

### ADR-008: DPO Algorithm Choice
- **Context**: Preference alignment algorithm selection
- **Options**: PPO/RLHF, DPO, KTO, ORPO, IPO, SimPO
- **Decision**: Standard DPO (Direct Preference Optimization)
- **Rationale**: PPO requires training a separate reward model (expensive, training-unstable). DPO directly uses preference pairs to optimize the policy relative to a frozen reference model — no RM needed. Our data is naturally paired (Apriori output as chosen, LLM failure as rejected) which DPO is designed for. TRL `DPOTrainer` is mature and well-tested. DPO loss is a straightforward cross-entropy variant — interpretable and stable.
- **Trade-offs**: DPO can be unstable if chosen/rejected are too similar in probability; requires paired data (we have it); can't optimize arbitrary reward signals
- **Source**: `src/training/export_real_dpo_data.py`; `notebooks/training_3phase_7b.ipynb` DPO section

### ADR-009: Real LLM Failures for DPO Rejected Outputs
- **Context**: How to generate "rejected" completions for DPO pairs
- **Options**: Synthetic corruption (random flip/delete items), model self-samples at high temperature, real LLM failures from pipeline runs
- **Decision**: Real LLM failures collected from `pipeline.py` runs on 4 models
- **Rationale**: 99.5% of real errors are `item_missing_in_row` — the model confidently cites an item as appearing in a row where it doesn't exist (hallucinated evidence). Random corruptions don't match this distribution — they would produce arbitrary errors that the model doesn't actually make. Training on real error patterns teaches the model to avoid its actual failure modes.
- **Trade-offs**: Requires running a costly LLM pipeline to collect failures; failure modes are partially model-specific
- **Source**: `src/training/export_real_dpo_data.py:11-14` (docstring with error distribution); `pipeline.py:64-141` (validation that flags these)

### ADR-010: GRPO Skipped in v3
- **Context**: Phase 3 training decision — run GRPO or skip?
- **Options**: Run GRPO as phase 3, skip GRPO
- **Decision**: Skip GRPO in v3
- **Rationale**: Council analysis (2026-03-09) recommended establishing a solid SFT+DPO baseline before adding GRPO. Designing a correct reward function for itemset extraction is non-trivial (F1 is non-differentiable; approximate rewards risk over-optimization). Risk of GRPO degrading a good DPO checkpoint. Decision: evaluate SFT+DPO first, then decide on GRPO in v4 if needed.
- **Trade-offs**: May miss additional F1 improvement from GRPO
- **Source**: `obsidian-brain/Experiments/2026-03-09 Council v3 Plan.md`; GRPO cells in notebook are present but skipped

### ADR-011: CoT with `<think>` Tags
- **Context**: Reasoning format for SFT training
- **Options**: Direct JSON output (no CoT), scratchpad comment, XML tags, Qwen-native `<think>` tags
- **Decision**: `<think>...</think>` tags followed by JSON
- **Rationale**: Qwen2.5-Instruct has native pre-training support for `<think>` tags (part of the model's instruct tuning). Explicitly separates reasoning trace from structured answer — enables ThinkStoppingCriteria during inference (stop at `</think>`, regenerate JSON at lower temperature). This separation is the key mechanism enabling two-phase inference (ADR-016).
- **Trade-offs**: Longer context usage per example; model must learn tag format exactly
- **Source**: `src/training/training_utils.py:22-34` (system prompt with `<think>` instruction); `src/evaluation/inference_utils.py` (ThinkStoppingCriteria)

### ADR-012: Column-Grouped CoT Format (v3)
- **Context**: How to structure step-by-step reasoning within `<think>` tags
- **Options**: Row-by-row scanning (natural reading order), flat item list, column-grouped scanning (chosen)
- **Decision**: Column-grouped scanning with RESULT SUMMARY termination signal
- **Rationale**: Row-by-row scanning causes repetition loops — the model encounters the same items across similar rows and gets stuck regenerating the same analysis. Column-grouped format organizes all occurrences of each item together (e.g., all `age:young` rows before moving to `age:old`), eliminating the repetition trigger. RESULT SUMMARY line provides a clear termination signal (`N singles + M pairs + K triples = TOTAL`). ~60% token reduction vs v2 row-by-row format. Council recommendation (2026-03-09).
- **Trade-offs**: Column-grouped is less intuitive than row-by-row; requires column-aware data loading in the CoT generator
- **Source**: `src/training/training_utils.py:60-191`; `obsidian-brain/Experiments/2026-03-09 Council v3 Plan.md`

### ADR-013: SFT Hyperparameters
- **Context**: Choosing the full SFT training configuration (v3 values, changed from v2)
- **Key decisions**:
  - `lr=1e-4`: Standard LoRA SFT starting point; **reduced from 2e-4** in v2 to prevent overfitting on 348 examples
  - `3 epochs`: **Increased from 2** in v2; more passes help given small dataset size
  - `batch=2, accum=4`: Memory constraint on T4/A100 with seq_len=2048; effective batch=8
  - `warmup=0.10`: **Increased from 0.05** in v2; larger warmup stabilizes training on small datasets
  - `weight_decay=0.01`: **Added in v3**; L2 regularization
  - `seq_len=2048`: **Reduced from 4096** in v2; v3 concise CoT fits comfortably within 2048
- **Source**: `notebooks/training_3phase_2026-03-09_v3.ipynb` Cell 2 (all values with change comments)

### ADR-014: DPO Hyperparameters
- **Context**: Choosing DPO training configuration
- **Key decisions**:
  - `lr=5e-5`: Lower than SFT — DPO loss is more sensitive to learning rate; standard literature recommendation
  - `β=0.1`: KL penalty weight in DPO loss; higher β → more conservative (stays closer to SFT policy). 0.1 is the TRL default and well-validated across use cases. We want some freedom to move away from the SFT policy but not too much.
  - `1 epoch`: **Reduced from 2** in v2; DPO can overfit quickly on 606 pairs — the loss can go to zero without true generalization
  - `batch=1, accum=4`: DPO processes chosen+rejected pairs together per step, effectively doubling memory use vs SFT at same settings
- **Source**: `notebooks/training_3phase_2026-03-09_v3.ipynb` Cell 2

### ADR-015: Sequence Length 2048
- **Context**: max_seq_length for training (was 4096 in v2)
- **Options**: 1024, 2048 (chosen), 4096 (v2), 8192
- **Decision**: 2048 in v3
- **Rationale**: v3 concise CoT format generates ~800–1200 tokens per example (estimated from `est_tokens` metadata in sft_cot_v3.json). A 2048 token budget provides 70%+ headroom. Longer sequences scale quadratically in attention memory — halving from 4096 to 2048 saves ~4× attention VRAM. The `--max-tokens 3500` filter in SFT data generation ensures no example exceeds budget.
- **Trade-offs**: Examples that were borderline under 4096 are now excluded (filtered at generation time)
- **Source**: `src/training/generate_cot_sft_data.py:69` (max_tokens filter); `notebooks/training_3phase_2026-03-09_v3.ipynb` Cell 2

### ADR-016: Two-Phase Inference
- **Context**: Inference strategy for the fine-tuned model at evaluation/deployment time
- **Options**: Single-pass generation, two-phase with stopping criterion, beam search, constrained decoding
- **Decision**: Two-phase generation — Phase 1: thinking at temp=0.3 + ThinkStoppingCriteria; Phase 2: JSON regeneration at temp=0.05
- **Rationale**: Single-pass generates `<think>+JSON` in one call but the JSON quality is sensitive to the thinking temperature — high temp for think = noisy JSON; low temp for think = repetition loops. Stopping at `</think>` and regenerating JSON at near-zero temperature decouples these tradeoffs. RepetitionDetector catches infinite thought loops before they hit the token limit.
- **Trade-offs**: 2× generation calls; requires custom StoppingCriteria; more complex inference pipeline
- **Source**: `src/evaluation/inference_utils.py:159-245`

### ADR-017: Thinking Temperature 0.3
- **Context**: Temperature value for the reasoning/CoT generation phase
- **Options**: greedy 0.0, 0.1, 0.3 (chosen), 0.5, 0.7+
- **Decision**: temp=0.3 for thinking phase
- **Rationale**: temp=0.0 (greedy) and temp=0.1 (near-greedy) lock the model into deterministic attractor states — the model repeatedly generates the same reasoning step in an infinite loop. temp=0.3 adds sufficient stochasticity to escape these attractors while keeping the reasoning focused. Council finding: <3% F1 degradation vs higher temperatures — acceptable cost for loop elimination.
- **Trade-offs**: Slight non-determinism in reasoning trace; rare cases where temp=0.3 generates slightly wrong CoT but still produces correct JSON
- **Source**: `src/evaluation/inference_utils.py:177-179`; `obsidian-brain/Experiments/` Council analysis

### ADR-018: Hash-Based Artifact Naming
- **Context**: How to name the JSON output files from pipeline.py runs
- **Options**: Timestamp-based (datetime in filename), sequential counter, hash-based (chosen)
- **Decision**: First 12 characters of SHA-256 of the CSV file content
- **Rationale**: Timestamp-based creates duplicates when re-running the same dataset. Sequential counters require a global state. Hash is fully deterministic — the same CSV file always produces the same artifact name → re-running is idempotent and doesn't pollute the artifacts directory. Enables deduplication across runs.
- **Trade-offs**: Artifact names are opaque (can't read model/dataset from filename); negligible collision probability at 500 files
- **Source**: `pipeline.py:176-181` (SHA-256 hash computation); `pipeline.py:36-46` (naming pattern)

### ADR-019: SQLite Persistence
- **Context**: How to persist pipeline run metadata across 1600+ runs
- **Options**: CSV logs, Parquet files, PostgreSQL, SQLite (chosen), no persistence
- **Decision**: SQLite (`runs.db`)
- **Rationale**: Single-file database — no server setup, no network dependency. Python's `sqlite3` is standard library — zero additional dependencies. Sufficient for 1600+ rows with complex queries (e.g. `WHERE validation_passed=0 AND llm_itemset_count > 0` for DPO selection). Gitignored → no privacy/size concerns. Dynamic schema (column addition via `ALTER TABLE`) allows evolving the schema without migration scripts.
- **Trade-offs**: Single-writer (fine for sequential pipeline); not portable for cloud/distributed use
- **Source**: `pipeline.py:252-336` (26-column schema, `persist_run_to_sqlite`)

### ADR-020: Adapter-Only Model Push
- **Context**: How to publish the fine-tuned model to HuggingFace Hub
- **Options**: Merge LoRA into base and push full weights (~14GB), push LoRA adapter only (~65MB), push GGUF quantized
- **Decision**: Push LoRA adapter weights only
- **Rationale**: `merged_4bit_forced` — Unsloth's method to merge a LoRA into a 4-bit NF4 base — destroys the adapter structure. The merge operation with NF4 quantization produces weights that are not cleanly reproducible. Adapter-only push is exact and deterministic: `base model + adapter = exact reproduction`. Users load `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` then apply the adapter — same result every time.
- **Trade-offs**: Users need to load base model separately; requires Unsloth or PEFT for inference
- **Source**: `notebooks/training_3phase_7b.ipynb` Cell 20 (comment about merged_4bit_forced failure); `obsidian-brain/Decisions/`

### ADR-021: min_support=3, max_size=3
- **Context**: Default Apriori parameters for the pipeline
- **Options**: Various combinations; min_support=2 (too many), 3 (chosen), 4 (too few for small sets); max_size=2, 3 (chosen), 4
- **Decision**: min_support=3, max_size=3
- **Rationale**: For datasets with 5–25 rows: min_support=2 causes combinatorial explosion (nearly every pair qualifies); min_support=4 on a 5-row dataset prunes everything. min_support=3 is the sweet spot — meaningful patterns without noise. max_size=3 keeps itemsets interpretable and computationally feasible; size-4 combinations are extremely rare in 3–15 column datasets and add negligible information.
- **Source**: `pipeline.py:595-596` (argparse defaults); `apriori_frequent_itemsets` function

### ADR-022: 13 Validation Invariants
- **Context**: How thoroughly to validate LLM output correctness
- **Options**: JSON parse check only, count check, schema validation, full semantic validation (chosen)
- **Decision**: 13 invariant checks spanning parse, structure, canonicalization, and semantic correctness
- **Rationale**: JSON parse alone misses semantic errors. The 13 invariants catch: JSON parseability, itemset array structure, non-empty items, canonical lowercase+sorted format, Row-N evidence format, support count verification against CSV, item existence in CSV (catches item_missing_in_row — the dominant failure mode, 99.5% of DPO rejected examples). The 13 invariants were determined empirically by observing all failure modes encountered in 1600+ runs.
- **Source**: `pipeline.py:64-141` (validate_all and individual check functions)

### ADR-023: Seven Evaluation Metrics
- **Context**: What to measure to assess model quality
- **Options**: Single accuracy score, F1 only, comprehensive multi-metric suite (chosen)
- **Decision**: 7 metrics: Precision, Recall, F1, Exact Match, Count Accuracy ±1, Hallucination Rate, JSON Parse Rate
- **Rationale**: F1 alone masks qualitatively different failure modes. Precision and Recall separately distinguish over-prediction from under-prediction. Exact Match is the gold standard for deployment readiness. Hallucination Rate specifically tracks the dominant failure mode identified in DPO data (item not in CSV). JSON Parse Rate tracks structural compliance — a model with good F1 but 50% parse rate is unusable in production.
- **Source**: `src/evaluation/eval_finetuned_model.py:417-500`

### ADR-024: Fixed Evaluation Set
- **Context**: How to select evaluation data across model comparisons
- **Options**: Random sampling at eval time, fixed holdout set (chosen)
- **Decision**: Fixed evaluation set (`data/eval_datasets_v2/`) — versioned, never changed
- **Rationale**: Random sampling means each evaluation run sees different data → models can't be compared fairly. A fixed set enables exact apples-to-apples comparison. Datasets are explicitly excluded from training by filename matching against `sft_cot_v2.json` and `dpo_real_v2.json`. The fixed set covers the full difficulty range (5–25 rows, 3–15 cols) to avoid bias.
- **Source**: `src/data_generation/generate_eval_datasets_v2.py`; `eval_finetuned_model.py:506-526`

### ADR-025: 500 Synthetic Datasets
- **Context**: Whether to use real-world or synthetic training data
- **Options**: Real-world transaction datasets (retail, e-commerce), synthetic data (chosen), hybrid
- **Decision**: 500 synthetic datasets generated with `src/data_generation/generate_datasets_v2.py`
- **Rationale**: Real-world datasets require licensing, curation, PII review, and cleaning. Synthetic gives full distribution control: row count 5–25, column count 3–15, item density configurable. Ensures training covers the full difficulty range uniformly. Reproducible — same generator seed → same 500 datasets. Teaches a generalizable skill, not dataset-specific patterns.
- **Trade-offs**: Synthetic item labels (random strings) don't match natural language item names; potential generalization gap to real-world data
- **Source**: `src/data_generation/generate_datasets_v2.py`; dataset naming `ds_<NNNN>_<rows>x<cols>.csv`

### ADR-026: Agent Workflow
- **Context**: How the project was developed and managed
- **Options**: Traditional single-developer scripts, manual pair programming, multi-agent AI workflow (chosen)
- **Decision**: 9-agent AI workflow with persistent memory (obsidian-brain)
- **Rationale**: The project has 8 clearly separable stages (organize → dataset → pipeline → training → deploy → pause → validate → finalize). Each stage benefits from a specialized agent with constrained scope. The obsidian-brain vault provides persistent memory across sessions — decisions, experiments, and bugs logged in structured markdown. Human defines success criteria (Apriori ground truth, 13 validation invariants, 7 metrics); agents execute under those guardrails.
- **Consequences**: All major decisions logged in `obsidian-brain/Decisions/`; experiment results in `obsidian-brain/Experiments/`; reproducible workflow state in `.github/agents_memory/workflow_state.json`
- **Source**: `.github/agents/` (9 agent definitions + 12 skill modules); `AGENTS.md`; `obsidian-brain/`

---

## 6. Reference Pages

### `docs/reference/hyperparameters.md`

Full table of every numeric parameter with source file:line:

| Parameter | Value | Phase | File:Line | Notes |
|---|---|---|---|---|
| `min_support` | 3 | Pipeline | `pipeline.py:595` | Apriori default |
| `max_size` | 3 | Pipeline | `pipeline.py:596` | Max itemset size |
| `chunk_size` | 50 | Pipeline | `pipeline.py:600` | Rows per LLM call |
| `llm_timeout` | 180s | Pipeline | `pipeline.py:486` | Per-request timeout |
| `llm_retries` | 2 | Pipeline | `pipeline.py:486` | On timeout |
| `llm_temperature` | 0.0 | Pipeline | `pipeline.py:483` | Extraction (not reasoning) |
| `val_ratio` | 0.10 | Data | `build_hf_dataset_v2.py:49,72,100` | 10% validation split |
| `seed` | 42 | Data | `build_hf_dataset_v2.py:49` | All splits + LoRA init |
| `max_sft_tokens` | 3500 | SFT data | `generate_cot_sft_data.py:69` | Per-example token budget |
| `max_cot_items` | 40 | SFT data | `generate_cot_sft_data.py:73` | Items shown in CoT |
| `max_dpo_rejected` | 3 | DPO data | `export_real_dpo_data.py:88` | Per dataset |
| `lora_r` | 32 | Training | `training_3phase_v3.ipynb Cell 2` | LoRA rank (was 64 in v2) |
| `lora_alpha` | 64 | Training | `training_3phase_v3.ipynb Cell 2` | α/r=2.0 ratio |
| `lora_dropout` | 0.05 | Training | `training_3phase_v3.ipynb Cell 2` | Added in v3 |
| `sft_lr` | 1e-4 | SFT | `training_3phase_v3.ipynb Cell 2` | Was 2e-4 in v2 |
| `sft_batch` | 2 | SFT | `training_3phase_v3.ipynb Cell 2` | Per device |
| `sft_accum` | 4 | SFT | `training_3phase_v3.ipynb Cell 2` | Effective batch=8 |
| `sft_epochs` | 3 | SFT | `training_3phase_v3.ipynb Cell 2` | Was 2 in v2 |
| `sft_warmup` | 0.10 | SFT | `training_3phase_v3.ipynb Cell 2` | Was 0.05 in v2 |
| `sft_weight_decay` | 0.01 | SFT | `training_3phase_v3.ipynb Cell 2` | Added in v3 |
| `sft_seq_len` | 2048 | SFT | `training_3phase_v3.ipynb Cell 2` | Was 4096 in v2 |
| `dpo_lr` | 5e-5 | DPO | `training_3phase_v3.ipynb Cell 2` | |
| `dpo_beta` | 0.1 | DPO | `training_3phase_v3.ipynb Cell 2` | KL penalty weight |
| `dpo_batch` | 1 | DPO | `training_3phase_v3.ipynb Cell 2` | Per device |
| `dpo_accum` | 4 | DPO | `training_3phase_v3.ipynb Cell 2` | Effective batch=4 |
| `dpo_epochs` | 1 | DPO | `training_3phase_v3.ipynb Cell 2` | Was 2 in v2 |
| `think_temp` | 0.3 | Inference | `inference_utils.py:177` | Escape attractor loops |
| `json_temp` | 0.05 | Inference | `inference_utils.py:179` | Near-greedy JSON |
| `max_gen_tokens` | 6000 | Inference | `inference_utils.py:115` | Hard cap |

### `docs/reference/database-schema.md`

Full `runs` table schema (26 columns) from `pipeline.py:252-336`:

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `timestamp` | TEXT | ISO 8601 run timestamp |
| `dataset_id` | TEXT | `<name>:<hash>` format |
| `dataset_hash` | TEXT | SHA-256[:12] of CSV |
| `min_support` | INTEGER | Apriori min_support used |
| `max_size` | INTEGER | Apriori max itemset size |
| `llm_full` | INTEGER | 1 if LLM extraction was run |
| `llm_chunk_size` | INTEGER | Rows per LLM chunk |
| `llm_model` | TEXT | Model ID (e.g. gpt-4.1-mini) |
| `apriori_itemset_count` | INTEGER | Ground truth count |
| `llm_itemset_count` | INTEGER | LLM output count |
| `validation_passed` | INTEGER | 0/1 |
| `apriori_valid_ratio` | REAL | Fraction passing validation |
| `llm_valid_ratio` | REAL | Fraction passing validation |
| `apriori_errors` | TEXT | JSON list of error types |
| `llm_errors` | TEXT | JSON list of error types |
| `apriori_duration_s` | REAL | Seconds |
| `llm_duration_s` | REAL | Seconds |
| `validation_duration_s` | REAL | Seconds |
| `total_duration_s` | REAL | Seconds |
| `n_rows` | INTEGER | Dataset row count |
| `n_cols` | INTEGER | Dataset column count |
| `artifact_path` | TEXT | Path to JSON output |
| `llm_artifact_path` | TEXT | Path to LLM JSON output |
| `dataset_path` | TEXT | Path to CSV |
| `notes` | TEXT | Free text |

### `docs/reference/prompt-templates.md`

Verbatim prompts used throughout:

1. **Training system prompt** (`src/training/training_utils.py:22-34`) — paste full text
2. **Pipeline extraction user message** (`pipeline.py:494-500`) — paste template
3. **Evaluation user message** (`eval_finetuned_model.py:324-329`) — paste template
4. **CoT format v3 example** — show a small worked example of column-grouped CoT

### `docs/reference/code-reference.md`

Use mkdocstrings to auto-generate from source:

```markdown
# Code Reference

## pipeline

::: pipeline

## src.training.generate_cot_sft_data

::: src.training.generate_cot_sft_data

## src.training.export_real_dpo_data

::: src.training.export_real_dpo_data

## src.evaluation.eval_finetuned_model

::: src.evaluation.eval_finetuned_model

## src.evaluation.inference_utils

::: src.evaluation.inference_utils
```

---

## 7. AI Workflow Pages

### `docs/ai-workflow/overview.md`

**The narrative: AI-augmented development done right.**

1. **Opening**: This project was developed using a structured multi-agent AI workflow. 9 specialized agents (defined in `.github/agents/`) executed specific tasks under a human-designed orchestration system with persistent memory in `obsidian-brain/`.

2. **The key principle**: AI handles execution velocity; human handles design and verification. Specifically:
   - Human designed: Apriori oracle (the correctness criterion), 13 validation invariants (the guardrails), 7 evaluation metrics (the success definition), agent architecture (the workflow)
   - Agents executed: data generation, pipeline runs, training data export, notebook generation, HF deployment, evaluation
   - Human reviewed: all eval results, all ADRs, all experiment outcomes, every version decision

3. **Why this is defensible**: The Apriori oracle is an immutable, mathematical ground truth. No matter what the agents produced, correctness was measured against an algorithm — not against AI judgment. This is the architectural key: self-validating through mathematics.

4. **The 8-stage workflow** (table: stage → agent → memory location)

5. **Quality control mechanisms**:
   - `runs.db` captures every run with validation_passed flag
   - 13 invariants in `pipeline.py` cannot be bypassed
   - LLM Council multi-model analysis validates major decisions
   - Experiment logs timestamped before training runs

6. **Comparison to standard practice**: Using AI agents for execution is analogous to using GitHub Copilot at scale — it accelerates routine coding tasks while the developer retains architectural and design authority.

### `docs/ai-workflow/agent-system.md`

Table of all 9 agents:

| Agent | Stage | Responsibilities | Memory |
|---|---|---|---|
| Orchestrator | All | Coordinates workflow, resolves conflicts | `obsidian-brain/Agents/Orchestrator.md` |
| Pipeline Agent | Stage 3 | Runs `pipeline.py` across 500 datasets | `obsidian-brain/Agents/Pipeline Agent.md` |
| Dataset Agent | Stage 2 | Generates 500 synthetic CSVs | `obsidian-brain/Agents/Dataset Agent.md` |
| Training Agent | Stage 4 | Exports training data, generates notebook | `obsidian-brain/Agents/Training Agent.md` |
| Deployment Agent | Stage 5 | Pushes to HuggingFace Hub | `obsidian-brain/Agents/Deployment Agent.md` |
| Evaluation Agent | Post-training | Runs eval, collects metrics | `obsidian-brain/Agents/Evaluation Agent.md` |
| Monitoring Agent | Stage 7 | Visualizes results, generates reports | `obsidian-brain/Agents/Monitoring Agent.md` |
| Cleanup Agent | Ongoing | Removes stale artifacts | `obsidian-brain/Agents/Cleanup Agent.md` |
| Maintainer Agent | Ongoing | Maintains repo, fixes bugs | `obsidian-brain/Agents/Maintainer Agent.md` |

Then explain the skill modules system: 12 modular skills in `.github/agents/skills/` that agents call.

### `docs/ai-workflow/experiment-journal.md`

Convert the 4 experiment logs from `obsidian-brain/Experiments/` into a chronological journal:

1. **2026-03-01: v1 Training Run** — first fine-tuning, baseline results, issues found
2. **2026-03-07: v2 Training Run** — improved SFT data, what changed from v1
3. **2026-03-08: v2 Adapter Evaluation** — LoRA adapter eval findings, repetition loop discovery
4. **2026-03-09: Council v3 Plan** — multi-model council analysis, all v3 changes decided (column-grouped CoT, r=32, GRPO skip)

For each entry: date, what was run, what was found, what decision it led to, link to relevant ADR.

---

## 8. `docs/glossary.md`

Terms to define (alphabetical):

- **Adapter** — LoRA weight matrices that extend a frozen base model
- **Apriori** — Classic frequent itemset mining algorithm; deterministic oracle in this project
- **β (DPO beta)** — KL penalty weight in DPO loss; controls how far the trained policy can deviate from SFT reference
- **Chain-of-Thought (CoT)** — Step-by-step reasoning trace before the final answer
- **DPO** — Direct Preference Optimization; preference alignment without a reward model
- **Evidence rows** — The "Row N" citations proving an itemset meets the support threshold
- **Frequent Itemset** — A set of items co-occurring in at least min_support transactions
- **GRPO** — Group Relative Policy Optimization; reinforcement learning variant (skipped in v3)
- **Hallucination (in LLM context)** — Citing an item in a row where it does not appear (`item_missing_in_row`)
- **LoRA** — Low-Rank Adaptation; efficient fine-tuning by training small rank-decomposed weight matrices
- **min_support** — Minimum number of transactions an itemset must appear in to be "frequent"
- **NF4** — Normal Float 4; 4-bit quantization format optimal for normally-distributed weights
- **Oracle** — A deterministic, authoritative source of ground-truth labels
- **QLoRA** — Quantized LoRA; LoRA applied to a 4-bit quantized base model
- **Repetition loop** — Failure mode where the model generates the same reasoning step indefinitely
- **SFT** — Supervised Fine-Tuning; training on human (or oracle) demonstrations
- **Support (count)** — The number of transactions containing all items in an itemset
- **Unsloth** — Training framework providing 2× speed and 70% VRAM reduction for LoRA fine-tuning
- **VRAM** — Video RAM; GPU memory used for model weights, activations, and optimizer states

---

## 9. Files to Create During Docs Writing

```
mkdocs.yml                                    ← Copy from Section 2 above
.github/workflows/docs.yml                   ← Copy from Section 3 above
docs/javascripts/mathjax.js                  ← Copy from Section 3 above
docs/index.md
docs/quickstart.md
docs/methodology/overview.md
docs/methodology/data-generation.md
docs/methodology/sft-training.md
docs/methodology/dpo-training.md
docs/methodology/evaluation.md
docs/decisions/index.md                      ← Table linking all 26 ADRs
docs/decisions/adr-001-apriori-as-oracle.md
docs/decisions/adr-002-qwen2.5-7b-selection.md
docs/decisions/adr-003-lora-vs-full-ft.md
docs/decisions/adr-004-lora-rank-32.md
docs/decisions/adr-005-4bit-nf4.md
docs/decisions/adr-006-unsloth.md
docs/decisions/adr-007-sft-before-dpo.md
docs/decisions/adr-008-dpo-not-ppo.md
docs/decisions/adr-009-real-failures-dpo.md
docs/decisions/adr-010-grpo-skipped.md
docs/decisions/adr-011-cot-think-tags.md
docs/decisions/adr-012-column-grouped-cot.md
docs/decisions/adr-013-sft-hyperparams.md
docs/decisions/adr-014-dpo-hyperparams.md
docs/decisions/adr-015-seq-length-2048.md
docs/decisions/adr-016-two-phase-inference.md
docs/decisions/adr-017-think-temp-0.3.md
docs/decisions/adr-018-hash-artifact-naming.md
docs/decisions/adr-019-sqlite.md
docs/decisions/adr-020-adapter-only-push.md
docs/decisions/adr-021-support-and-size.md
docs/decisions/adr-022-validation-invariants.md
docs/decisions/adr-023-eval-metrics.md
docs/decisions/adr-024-fixed-eval-set.md
docs/decisions/adr-025-synthetic-datasets.md
docs/decisions/adr-026-agent-workflow.md
docs/reference/hyperparameters.md
docs/reference/database-schema.md
docs/reference/prompt-templates.md
docs/reference/code-reference.md
docs/ai-workflow/overview.md
docs/ai-workflow/agent-system.md
docs/ai-workflow/experiment-journal.md
docs/glossary.md
CITATION.cff                                 ← See Section 10
obsidian-brain/README.md                     ← Explain the vault (Option B)
```

---

## 10. `CITATION.cff`

```yaml
cff-version: 1.2.0
message: "If you use this work, please cite it using the following metadata."
title: "Itemset Extraction via Fine-tuned Qwen2.5-7B: An Apriori-Oracle Approach"
authors:
  - family-names: Slivka
    given-names: Oliver
    affiliation: "Faculty of Informatics and Statistics, Prague University of Economics and Business (VŠE)"
type: software
version: 1.0.0
date-released: "2026-04-XX"  # fill in actual release date
repository-code: https://github.com/YOUR_USERNAME/itemsety-qwen-finetuning
license: MIT
abstract: >
  An end-to-end ML pipeline fine-tuning Qwen2.5-7B for frequent itemset extraction.
  Uses Apriori as a deterministic ground-truth oracle, eliminating human annotation.
  Training strategy: SFT with chain-of-thought reasoning (348 examples) followed by
  DPO on real LLM failure examples (606 pairs). Achieves F1=? on a fixed evaluation set.
keywords:
  - machine-learning
  - fine-tuning
  - frequent-itemsets
  - lora
  - dpo
  - qwen
  - nlp
```

---

## 11. Writing Order (Recommended)

Write in this order for maximum efficiency:

1. `mkdocs.yml` + GitHub Actions workflow + `docs/javascripts/mathjax.js` → deploy empty skeleton → verify Pages live
2. `docs/index.md` → the landing page sets the tone for everything else
3. `docs/methodology/overview.md` → the Mermaid diagram clarifies the story
4. `docs/reference/hyperparameters.md` → all numbers in one place, easy to reference while writing ADRs
5. All 26 ADRs (`docs/decisions/`) → write them top to bottom, they're the load-bearing piece
6. `docs/decisions/index.md` → summary table of all ADRs
7. `docs/methodology/` sub-pages → draw from methodology/overview.md and ADRs
8. `docs/ai-workflow/` → write after methodology, reference the agents
9. `docs/reference/database-schema.md`, `prompt-templates.md`, `code-reference.md`
10. `docs/quickstart.md`, `docs/glossary.md`
11. `CITATION.cff` + `obsidian-brain/README.md`
12. Final pass: fill in all `?` placeholders with actual eval numbers
13. `mkdocs gh-deploy`
