# 2026-03-08 Training Agent — Adapter Eval Diagnostic Session

**Agent:** Training Agent  
**Date:** 2026-03-08  
**Duration:** ~4 hours (multi-session)  
**Status:** ⚠️ Partial — diagnostic complete, model needs v3 training  
**Tags:** #log #agent/training #adapter-eval #diagnostic

---

## Summary
Continued adapter-only evaluation from yesterday's council diagnostic. Created raw capture eval script (no penalties) → discovered repetition loops as root cause → created repetition penalty eval script → discovered `no_repeat_ngram_size=3` destroys structured output → concluded model is undertrained and needs v3 training.

## Actions
1. Created `notebooks/eval_raw_capture.py` — standalone eval with max_new_tokens=8192, saves full raw output per dataset
2. User ran on TLJH: 15 datasets, no penalties → 2/15 success (F1=78.6%, 33.8%), 13/15 repetition loops
3. Analyzed raw outputs — identified 3 repetition failure modes (think-line repeat, JSON row spam, hallucinated numbering)
4. Created `notebooks/eval_with_reppenalty.py` — same eval with repetition_penalty=1.3, no_repeat_ngram_size=3
5. User ran on TLJH: 30 datasets → 0/30 success, 0% parse rate (penalties garbled all output)
6. Analyzed reppenalty results — `no_repeat_ngram_size=3` is catastrophic for JSON/template output
7. Updated Training Agent memory with both findings
8. Created experiment note: `obsidian-brain/Experiments/2026-03-08 Qwen2.5-7B v2 Adapter Eval.md`

## Results
- **Raw capture (no penalty):** 2/15 completed, avg F1=7.5%, best F1=78.6% on 10×3 dataset
- **With repetition penalty:** 0/30 parseable, avg F1=0.0%, all output garbled
- **Conclusion:** Model is undertrained (314 SFT examples too few for termination). Inference params can't fix this.

## Artifacts Created
- `notebooks/eval_raw_capture.py` — raw capture eval script
- `notebooks/eval_with_reppenalty.py` — repetition penalty eval script  
- `eval_raw_capture/` — 15 raw outputs + summary.json (on local + TLJH)
- `eval_reppenalty/` — 30 raw outputs + summary.json (on local + TLJH)
- `eval_reppenalty.tar.gz` — compressed results from TLJH
- `obsidian-brain/Experiments/2026-03-08 Qwen2.5-7B v2 Adapter Eval.md`
- `obsidian-brain/Agents/Training Agent.md` — updated with both diagnostic findings

## Issues & Notes
- `no_repeat_ngram_size` must NEVER be used for structured output (JSON, templates)
- `repetition_penalty=1.3` is too aggressive for this model — garbles text
- The v2 adapter WORKS on small datasets — proves training partially succeeded
- Next session: LLM Council with full diagnostic data to decide v3 training plan

---

**Related:** [[Experiments/2026-03-08 Qwen2.5-7B v2 Adapter Eval]], [[Agents/Training Agent]]
