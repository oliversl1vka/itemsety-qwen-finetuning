# Deployment Agent Memory

Persistent knowledge store for HuggingFace Hub deployment insights.

**Agent file:** `.github/agents/deployment-agent.md`  
**Tags:** #agent/deployment

---

<!-- Append new memories below, newest last -->

## 2026-03-01 — Token Export Pattern

**Issue:** `source hf.env` does NOT propagate `HF_TOKEN` to child Python processes when chained with `&&`.  
**Fix:** Always use `export $(grep -v '^#' hf.env | xargs)` before running upload scripts.  
**In code:** Call `login(token=os.environ['HF_TOKEN'])` at the top of `main()`, and pass `token=` explicitly to both `push_to_hub()` and `HfApi()`.  
**Also:** Repo must exist before uploading — use `create_repo(repo, repo_type='dataset', exist_ok=True, token=token)`.  
**Related log:** [[2026-03-01_deployment_push]]

---

## 2026-03-07 — huggingface-cli Not in .venv

**Issue:** `huggingface-cli` command not found inside `.venv` even though `huggingface_hub` is installed.  
**Fix:** Use Python API directly: `HfApi().upload_file(path_or_fileobj=..., path_in_repo=..., repo_id=..., repo_type='dataset')`.  
**Also works:** `HfApi().upload_folder()` for directories.  
**Application:** Always prefer Python API over CLI in this project.  
**Related log:** [[2026-03-07_deployment_push]]

---

## 2026-03-07 — v2 Push (Council-Corrected Notebook)

**Context:** Re-pushed after v1 training failure (0% F1). Uploaded v2 notebook (SFT→GRPO, no DPO) + refreshed dataset card with council-fixed params.  
**Files uploaded:** 3 dataset configs (sft/dpo/grpo), v2 notebook, 2 eval scripts = 16 total files in repo.  
**Key:** v1 notebook left in repo at `notebooks/training_3phase_7b.ipynb` for reference; v2 at `notebooks/training_3phase_2026-03-07_v2.ipynb`.  
**Verification:** All 3 configs load from Hub, all files accessible.  
**Tags:** #v2 #push #council-corrected

---

## 2026-03-09 — v3 Push (Concise CoT + DPO Re-enabled)

**Context:** Pushed v3 training data with concise CoT format (column-grouped, RESULT SUMMARY termination signal) after full LLM Council diagnostic of v2's 0.4% F1 failure. DPO re-enabled with real LLM failures.  
**Files uploaded:** 3 dataset configs (sft 233/25, dpo 546/60, grpo 233/25), v3 notebook, 2 eval scripts (eval_finetuned_model.py + inference_utils.py), 30 eval CSVs.  
**Key:** v1 and v2 notebooks preserved in repo for reference. v3 notebook at `notebooks/training_3phase_2026-03-09_v3.ipynb`.  
**Script used:** `scripts/push_v3_assets.py` for non-dataset files (HfApi.upload_file + upload_folder).  
**Verification:** All 3 configs loaded from Hub successfully, all individual files accessible.  
**No issues encountered.** Upload was smooth — no timeout, no token issues.  
**Tags:** #v3 #push #concise-cot #dpo-real

---

## [2026-03-17] 🔬 Diamond Knowledge — Export & Deployment Patterns

**Source:** Review of 11 Unsloth × Qwen notebooks (diamond phase extraction)

### GGUF Multi-Quant Export (One Call)

All 11 reviewed notebooks use multi-quantization in a single `push_to_hub_gguf` call:
```python
model.push_to_hub_gguf(
    "HF_USER/model-GGUF",
    quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
    token=os.environ["HF_TOKEN"],
)
```
This creates 3 GGUF variants simultaneously. Currently we push LoRA adapters only — GGUF is a future option for llama.cpp deployment.

### Preferred Merge Path: 16-bit

Diamond review confirms: `merged_16bit` is the cleanest merge path for adapter export.
- `merged_4bit` → **DANGEROUS** on NF4 models (quantization cascade, same bug we hit in v2/v3)
- `merged_16bit` → safe, works with VLLM, produces standard safetensors
- LoRA-only → smallest, requires base model at inference (our current approach — correct)

**NEVER use `merged_4bit_forced`** — this is the bug that destroyed v1, v2, and v3 pre-fix.

### TorchAO Export (Future Alternative to GGUF)

QAT notebooks show `save_pretrained_torchao()` with `Int4WeightOnlyConfig` — newer export path compatible with vLLM. Not applicable now (we don't use QAT) but worth noting for future if we explore QAT fine-tuning.

**Tags:** #diamond-knowledge #export #gguf #deployment-patterns

See also: [[Training Agent]] [[References/Unsloth Notebook Patterns]]

---

## [2026-03-18] ⚠️ HF Dataset Repos Are Now VERSIONED — Never Overwrite

**Context:** Since 2026-03-01, all dataset pushes overwrote `OliverSlivka/itemset-extraction-v2`, destroying training history. Fixed by creating separate repos per version.

**Current repos:**
| Version | Repo | Status |
|---------|------|--------|
| v3 (current) | `OliverSlivka/itemset-extraction-v3` | Active |
| v2 (frozen) | `OliverSlivka/itemset-extraction-v2` | 🔒 Never modify |

**CRITICAL RULE:** When pushing a NEW dataset version, ALWAYS create a new repo:
- `OliverSlivka/itemset-extraction-v{N}` (e.g., v4, v5, ...)
- **NEVER push to v2 or v3** — they are frozen historical snapshots
- Push script: `scripts/push_versioned_datasets.py`

See also: [[Decisions/HF Dataset Versioning 2026-03-18]] [[References/HF Dataset Repos]]

**Tags:** #versioning #critical #hf-repos
