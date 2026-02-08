# OpenAI API Token Limits

**Last Updated:** 2026-02-08  
**Tags:** #api #reference

---

## Tier 1 — 250k tokens/day (expensive models, shared pool)

| Model |
|-------|
| `gpt-5.2` |
| `gpt-5.1` |
| `gpt-5.1-codex` |
| `gpt-5` |
| `gpt-5-codex` |
| `gpt-5-chat-latest` |
| `gpt-4.1` |
| `gpt-4o` |
| `o1` |
| `o3` |

## Tier 2 — 2.5M tokens/day (mini/nano models, shared pool)

| Model |
|-------|
| `gpt-5.1-codex-mini` |
| `gpt-5-mini` |
| `gpt-5-nano` |
| `gpt-4.1-mini` |
| `gpt-4.1-nano` |
| `gpt-4o-mini` |
| `o1-mini` |
| `o3-mini` |
| `o4-mini` |
| `codex-mini-latest` |

## Planning Guidance

At ~2000 tokens per dataset extraction:
- **Tier 1:** ~125 datasets/day → multi-day batch for 500+ datasets
- **Tier 2:** ~1250 datasets/day → single-day batch for 500+ datasets

## Recommendation

Always prefer **Tier 2 models** for batch pipeline runs:
```bash
python pipeline.py --data-dir data/datasets_v2 --llm-full --llm-model gpt-4.1-mini
```

**⚠️ Do NOT plan batch runs with Tier 1 models** unless quality requires it.

---

**Related:** [[Agents/Pipeline Agent]], [[Agents/Training Agent]], [[References/Model Comparison]]
