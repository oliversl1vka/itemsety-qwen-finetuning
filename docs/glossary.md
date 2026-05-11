# Glossary

Domain-specific terms used throughout this project, in alphabetical order.

---

**Adapter**
:   A set of small, trainable weight matrices (LoRA) that extend a frozen base model. The adapter encodes the fine-tuned behavior; the base model weights are never modified. This project's adapter is ~65 MB vs the 14 GB base model.

**Apriori**
:   A classic algorithm for mining frequent itemsets from transactional data. Given a minimum support threshold, it enumerates all itemsets (up to a maximum size) that appear in at least that many transactions. In this project, Apriori serves as the deterministic **oracle** — its output is the ground truth against which all LLM outputs are evaluated.

**BitsAndBytes (bnb)**
:   A library for quantizing model weights to 4-bit or 8-bit precision. Used here for NF4 quantization of the base Qwen2.5-7B model, reducing VRAM requirements from ~14 GB to ~4 GB.

**Chain-of-Thought (CoT)**
:   A prompting and training technique where the model generates step-by-step reasoning before producing the final answer. This project uses a structured CoT format inside `<think>` tags with column-grouped scanning (singles, pairs, triples).

**ChatML**
:   The chat markup format used by Qwen models. Messages are structured as `<|im_start|>system\n...<|im_end|>`, `<|im_start|>user\n...<|im_end|>`, `<|im_start|>assistant\n...<|im_end|>`. Training data follows this format.

**DPO (Direct Preference Optimization)**
:   A preference alignment algorithm that trains a model to prefer "chosen" outputs over "rejected" outputs without requiring a separate reward model. The loss function directly optimizes the policy using paired examples. In this project, chosen = Apriori ground truth, rejected = real LLM failures.

**Evidence rows**
:   The `"Row N"` citations in model output that prove an itemset meets the minimum support threshold. For example, `"rows": ["Row 1", "Row 3", "Row 5"]` claims the itemset appears in those three transactions. The validation invariants verify each citation against the actual CSV data.

**F1 Score**
:   The harmonic mean of precision and recall. Computed at the itemset level: each predicted itemset is compared against Apriori's output. An itemset is a true positive if the exact same set of items appears in both predicted and ground-truth outputs.

**Frequent itemset**
:   A set of items that co-occur in at least `min_support` transactions. For example, if `{bread, milk}` appears together in 4 out of 10 transactions and `min_support=3`, then `{bread, milk}` is frequent with count=4.

**GRPO (Group Relative Policy Optimization)**
:   A reinforcement learning variant that generates multiple completions per prompt and uses relative rewards (within the group) to update the policy. Attempted in iteration 2 (200 steps, near-zero F1 signal, json_format≈0.95 — deemed counterproductive). Skipped in the final iteration (v3).

**Hallucination**
:   In this project, a hallucination is specifically an `item_missing_in_row` error — the model cites an item as appearing in a row where it does not exist in the CSV data. This is the dominant failure mode, accounting for 99.5% of DPO rejected examples.

**LoRA (Low-Rank Adaptation)**
:   An efficient fine-tuning technique that decomposes weight updates into two small matrices of rank `r`, rather than modifying the full weight matrix. This project uses r=32, alpha=64, targeting all attention (q, k, v, o) and MLP (gate, up, down) projections.

**min_support**
:   The minimum number of transactions in which an itemset must appear to be considered "frequent." Set to 3 in this project — the empirical sweet spot for datasets with 4-26 rows.

**NF4 (Normal Float 4)**
:   A 4-bit quantization format designed for normally distributed weights. More accurate than uniform 4-bit quantization because it allocates more precision near the distribution center where most weights cluster.

**Oracle**
:   A deterministic, authoritative source of ground-truth labels. In this project, the Apriori algorithm is the oracle — its output defines what is "correct" for any given dataset and minimum support value.

**primary_v3**
:   The canonical evaluation profile used for final model comparison. Uses two-phase CoT inference (think temperature 0.3, JSON temperature 0.05) with `ThinkStoppingCriteria` and `RepetitionDetector`. All headline metrics (F1=12.6% SFT, 11.8% DPO, 0% hallucination) are reported under this profile. Alternative profiles (`raw_capture`, `reppenalty`) exist for ablation.

**QLoRA (Quantized LoRA)**
:   LoRA applied to a quantized (4-bit) base model. The base weights are frozen in NF4 precision; only the LoRA adapter matrices are trained in bfloat16. This enables fine-tuning a 7B model on a single GPU.

**Repetition loop**
:   A failure mode where the model generates the same reasoning step or output fragment indefinitely, hitting the token limit without producing a final answer. Discovered in iteration 2 raw capture evaluation (87% = 13/15 datasets). Addressed in the final iteration via concise CoT format, `ThinkStoppingCriteria`, and `RepetitionDetector`.

**runs.db**
:   The central SQLite database that stores every pipeline run with 27 columns of metadata (timestamps, dataset identifiers, parameter settings, output counts, validation results, file paths). Both training data generation scripts query `runs.db`: SFT data selects one valid Apriori run per unique dataset; DPO data selects runs where `validation_passed=0 AND llm_itemset_count > 0` (real LLM failures). The database is gitignored but available as a release asset for reproducibility.

**SFT (Supervised Fine-Tuning)**
:   Training a model on demonstration examples — input/output pairs where the output is the desired behavior. In this project, SFT uses 272 examples where the input is a CSV dataset + min_support and the output is a structured CoT reasoning trace followed by the JSON itemset array.

**Support (count)**
:   The number of distinct transactions containing all items in an itemset. If items `{a, b}` both appear in rows 1, 4, and 7, the support count is 3.

**ThinkStoppingCriteria**
:   A custom `transformers.StoppingCriteria` that halts generation when the model produces a `</think>` token. Used in two-phase inference to separate the reasoning phase (temp=0.3) from the JSON output phase (temp=0.05). Source: `src/evaluation/inference_utils.py`.

**Two-phase inference**
:   The inference protocol used for fine-tuned models. Phase 1 generates reasoning inside `<think>` tags at temperature 0.3 until `ThinkStoppingCriteria` fires. Phase 2 then generates the structured JSON output at temperature 0.05. This decouples exploratory reasoning from precise structured output. See [ADR-016](decisions/adr-016-two-phase-inference.md).

**Unsloth**
:   A training framework that provides 2x speed and 70% VRAM reduction for LoRA fine-tuning through optimized attention kernels and memory management. Used for both SFT and DPO training in this project.

**VRAM (Video RAM)**
:   GPU memory used for model weights, activations, gradients, and optimizer states during training and inference. The QLoRA + Unsloth stack reduces peak VRAM usage to ~6-7 GB for a 7B model, fitting within a single GPU.
