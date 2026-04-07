# Dataset Agent Memory

Persistent knowledge store for dataset generation insights.

**Agent file:** `.github/agents/dataset-agent.md`  
**Tags:** #agent/dataset

---

## [2026-02-03] Optimized size distribution working well

**Context:** 500 datasets generated with optimized distribution.  
**Insight:**
- 47% small (5-10 rows), 35% medium (11-18 rows), 18% large (19-25 rows)
- Average token usage ~272 per dataset (well within LLM context limits)
- 10 valid source datasets loaded from `real_datasets/` (15 skipped for insufficient categorical columns)
- Variation strategy distribution: 27% row_subsample, 23% combined, 19% col_subsample, 16% shuffle, 15% noise

**Application:** All datasets fit within typical LLM context windows (32k-128k tokens). Current distribution is well-balanced.  
**Tags:** #insight #dataset-generation

---

## [2026-03-17] 🔬 Diamond Knowledge — Dataset & Formatting Patterns

**Source:** Review of 11 Unsloth × Qwen notebooks (diamond phase extraction)

### Prompt Length Filtering (90th Percentile Cutoff)

The DeepSeek-R1 notebook filters out the top 10% longest prompts:
```python
lengths = [len(tokenizer.encode(x)) for x in dataset["text"]]
cutoff = np.quantile(lengths, 0.9)
dataset = dataset.filter(lambda x: len(tokenizer.encode(x["text"])) <= cutoff)
```
**Applicability:** Our SFT data is already fairly uniform (~500-700 tokens after concise v3 CoT format). The max_seq_length=2048 provides ample headroom. **Not needed now**, but if we expand to longer datasets or add multi-turn examples, apply this filtering.

### Chat Template Confirmed: `"qwen-2.5"`

All Qwen2.5 notebooks use `get_chat_template(tokenizer, chat_template="qwen-2.5")`. Our notebook uses `tokenizer.apply_chat_template()` directly which auto-detects from the model config — equivalent and correct.

### `standardize_data_formats()` vs `standardize_sharegpt()`

Diamond review: `standardize_data_formats()` is the newer, more generic auto-detection function (preferred). `standardize_sharegpt()` is legacy for ShareGPT `{from/value}` format only. Our HF dataset already uses standard `{role/content}` format, so neither is needed — but if we ever load external datasets, use `standardize_data_formats()`.

### Packing Consideration for Short Sequences

Our CSV prompts are short (~500-700 tokens). `packing=True` concatenates multiple samples to fill `max_seq_length` → potential 5× throughput improvement. **However:** v3 Council explicitly set `packing=False` because packing can corrupt `train_on_responses_only` masking boundaries. **Decision:** Leave `packing=False` until we can validate masking integrity with packing enabled in a test run.

**Tags:** #diamond-knowledge #dataset-patterns #formatting

See also: [[Training Agent]] [[References/Unsloth Notebook Patterns]]
