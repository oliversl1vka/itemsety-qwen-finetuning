---
name: llm-itemset-extraction
description: Use Azure OpenAI to extract frequent itemsets from CSV data. Provides baseline LLM performance for comparison with fine-tuned models.
---

# LLM Frequent Itemset Extraction

Call Azure OpenAI (GPT-4) to extract frequent itemsets from CSV transactions.

## Overview

LLM extraction provides:
- Baseline performance metrics (before fine-tuning)
- Training data generation (LLM response as label)
- Comparison point for fine-tuned model evaluation

## Prerequisites

### Azure Credentials
Create `azure.env` from template:
```bash
cp azure.env.template azure.env
# Edit with your credentials:
# AZURE_OPENAI_API_KEY=your_key
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# AZURE_OPENAI_API_VERSION=2024-02-15-preview
# AZURE_OPENAI_DEPLOYMENT=gpt-4
```

## Quick Start

### Single Dataset (Full Mode)
```bash
python pipeline.py --data data/datasets_v2/ds_0001_15x10.csv --llm-full --llm-model gpt_4_1
```

### Batch with Chunking
```bash
python pipeline.py --data-dir data/datasets_v2 --llm-full --llm-model gpt_4_1 --llm-chunk-size 50
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--llm-full` | false | Enable full LLM extraction |
| `--llm-model` | gpt_4_1 | Model identifier for logging |
| `--llm-chunk-size` | 100 | Datasets per API batch |

## Prompt Structure

The system prompt instructs the LLM to:
1. Analyze CSV data for frequent patterns
2. Return JSON array of itemsets
3. Include count, rows, and support for each
4. Use strict format (no markdown, no explanation)

See `extractor_system_prompt.md` for full prompt.

## Expected Output Format

```json
[
  {
    "itemset": ["Apple", "Banana"],
    "count": 5,
    "rows": ["Row 1", "Row 3", "Row 5", "Row 8", "Row 10"],
    "support": 0.333
  }
]
```

## Output Artifacts

| File | Location |
|------|----------|
| LLM output | `artifacts/extractor_outputs/gpt_4_1_extractor_output_{stem}_{hash}.json` |
| Log | `logs/extractor/gpt_4_1_extractor_generation_log_{stem}_{hash}.json` |

## Rate Limiting

Azure OpenAI has rate limits. Mitigate with:

```bash
# Smaller chunks
--llm-chunk-size 25

# Add delay (in code)
time.sleep(2)  # Between calls
```

### Error Codes
- **429:** Rate limit exceeded — reduce chunk size, add delays
- **401:** Invalid credentials — check azure.env
- **404:** Deployment not found — verify deployment name

## Validation

LLM output is validated against 13 invariants:

1. JSON parseable
2. Array format (not object)
3. Each itemset has required fields
4. Count matches row list length
5. Items exist in original dataset
6. Row labels valid
7. Support correctly calculated
8. No duplicate items in itemset
9. Itemsets sorted alphabetically
10. No duplicate rows in evidence
11. Count >= min_support
12. Itemset size <= max_size
13. No hallucinated items

## Performance Targets

| Metric | Target | Current (GPT-4) |
|--------|--------|-----------------|
| JSON Parse Rate | 95%+ | ~85% |
| Precision | 80%+ | ~75% |
| Recall | 80%+ | ~70% |
| F1 Score | 80%+ | ~72% |
| Latency | <60s | ~45s |

## Troubleshooting

### Empty response
- Check prompt length (may exceed context)
- Verify dataset is not empty
- Check API credentials

### Invalid JSON
- LLM may include markdown (```json)
- Strip text before first `[` and after last `]`
- Increase temperature to 0

### Hallucinated items
- Items not in original dataset
- Common with complex datasets
- Validation catches these automatically
