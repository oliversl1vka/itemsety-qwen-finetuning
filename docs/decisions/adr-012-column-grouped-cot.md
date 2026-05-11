# ADR-012: Column-Grouped CoT Format

**Status:** Accepted  
**Date:** 2026-03-09 (v3 change)

## Context

The chain-of-thought reasoning within `<think>` tags must systematically scan the dataset for frequent items, pairs, and triples. How should this scan be structured?

## Options Considered

| Format | Token Cost | Repetition Risk | Clarity |
|--------|-----------|-----------------|---------|
| Row-by-row scanning (v2) | High | High -- same items revisited | Natural reading order |
| Flat item list | Low | Low | Hard to follow reasoning |
| **Column-grouped scanning (v3)** | Low (~40% less) | Low | Grouped by item |

## Decision

**Column-grouped scanning** with a RESULT SUMMARY termination signal.

## Rationale

The v2 row-by-row format caused severe repetition loops during inference. When scanning row by row, the model encounters the same items across similar rows and gets stuck regenerating the same analysis. For example, if "young" appears in rows 1, 2, 5, and 7, the row-by-row format would mention "young" four times in different contexts, creating attractor states for the generation.

Column-grouped format resolves this by organizing all occurrences of each item together:

```
age:young=4(R1,R2,R5,R7)✓
```

This structure:

1. **Eliminates repetition triggers** -- each item appears once with all its rows
2. **Reduces tokens by ~40%** -- compact notation vs. verbose row-by-row descriptions
3. **Provides RESULT SUMMARY** (`N singles + M pairs + K triples = TOTAL`) as a clear termination signal, preventing the model from continuing to generate after completing its analysis
4. **Mirrors Apriori's logic** -- the algorithm itself works item-by-item, not row-by-row

## Trade-offs

- Column-grouped scanning is less intuitive for human readers (rows are the natural unit)
- Requires column-aware data loading in the CoT generator
- The compact notation (`col:val=count(rows)✓`) is domain-specific and not transferable

## Source Evidence

- `src/training/training_utils.py:60-191` -- v3 CoT generator implementation
- `obsidian-brain/Experiments/2026-03-09 Council v3 Plan.md` -- Council recommendation
