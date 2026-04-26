# ADR-025: 500 Synthetic Datasets

**Status:** Accepted  
**Date:** 2026-02

## Context

The pipeline requires tabular CSV datasets for training and evaluation. Should these be real-world transaction datasets or synthetically generated?

## Options Considered

| Source | Control | Licensing | Privacy | Reproducibility |
|--------|---------|-----------|---------|----------------|
| Real-world (retail, e-commerce) | None | Requires licensing | PII risk | Depends on source |
| **Synthetic** | Full control | None needed | Zero risk | Fully reproducible |
| Hybrid | Partial | Partial | Moderate | Partial |

## Decision

**500 synthetic datasets** generated programmatically.

## Rationale

- **Full distribution control**: Row count (5--25), column count (3--15), and item density are all configurable. This ensures uniform coverage of the difficulty space.
- **Zero licensing friction**: No data use agreements, no attribution requirements, no commercial restrictions
- **Zero privacy risk**: No real customer data, no PII
- **Full reproducibility**: Same generator seed produces the same 500 datasets every time
- **Teaches a general skill**: The model learns the itemset extraction task abstractly, not dataset-specific patterns

## Trade-offs

- Synthetic item labels (random strings like "cat_A", "val_2") don't match natural language item names found in real transaction data
- The model may struggle to generalize to real-world datasets with natural language item descriptions
- No benchmark comparison against existing itemset mining literature (which uses real-world datasets)

## Source Evidence

- `src/data_generation/generate_datasets_v2.py` -- generator implementation
- Dataset naming convention: `ds_<NNNN>_<rows>x<cols>.csv`
