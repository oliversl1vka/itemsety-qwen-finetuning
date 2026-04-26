# ADR-018: Hash-Based Artifact Naming

**Status:** Accepted  
**Date:** 2026-02

## Context

The pipeline generates JSON artifact files for each run. How should these files be named to prevent conflicts and enable reproducibility?

## Options Considered

| Strategy | Idempotent | Readable | Collision Risk |
|----------|-----------|----------|---------------|
| Timestamp | No (different name each run) | Date visible | None |
| Sequential counter | No (requires global state) | Order visible | None |
| **SHA-256 hash of CSV** | Yes (same CSV = same name) | Opaque | Negligible |

## Decision

**First 12 characters of SHA-256 hash** of the CSV file content, combined with the model prefix and dataset stem.

Pattern: `<model_prefix><kind>_<dataset_stem>_<hash[:12]>.json`

## Rationale

Hash-based naming is **idempotent**: running the pipeline on the same CSV file always produces the same artifact filename. This means:

1. **Re-running is safe** -- repeated pipeline runs on the same dataset overwrite rather than creating duplicates
2. **Deduplication is automatic** -- no cleanup scripts needed
3. **No global state required** -- unlike sequential counters, no external counter file

At 12 hex characters (48 bits), the collision probability for 500 files is approximately \(5 \times 10^{-11}\) (birthday paradox), which is negligible.

## Trade-offs

- Artifact names are opaque -- cannot determine the model or dataset by reading the filename alone
- Requires computing SHA-256 for each CSV, though this is negligible compared to LLM inference time

## Source Evidence

- `pipeline.py:176-181` -- SHA-256 computation
- `pipeline.py:36-46` -- naming pattern construction
