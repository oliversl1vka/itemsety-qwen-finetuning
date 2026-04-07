# Monitoring Agent Memory

Persistent knowledge store for observability and reporting insights.

**Agent file:** `.github/agents/monitoring-agent.md`  
**Tags:** #agent/monitoring

---

<!-- Append new memories below, newest last -->

## [2026-03-17] 🔬 Diamond Knowledge — Monitoring Patterns

**Source:** Review of 11 Unsloth × Qwen notebooks (diamond phase extraction)

### VRAM Tracking Between Phases

All multi-phase notebooks print VRAM stats after cleanup:
```python
print(f"VRAM: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB free")
```
Already in our v3 Cell 9 ✅. When building monitoring dashboards, capture these values for each phase transition.

### Reward Distribution Monitoring (For Future GRPO)

GRPO notebooks show that 0 reward for the first 100+ steps is **normal** — the model is learning format. If reward stays 0 after 200+ steps, the reward functions are broken (this is exactly what happened in our v1 catastrophic failure).

**Key metrics to track if GRPO is re-enabled:**
- Reward per function (format vs correctness) — should see format reward rise first, then correctness
- KL divergence — should stay bounded; if it explodes, reduce `num_generations` or increase `beta`
- Generation diversity — if all `num_generations` samples are identical, temperature/top_k too low

### Training Loss Curves

Diamond review validated our tracking approach:
- SFT: loss should decrease steadily, val loss should follow (no overfitting gap)
- DPO: reward margin (chosen - rejected) should increase; accuracy should converge to ~1.0
- Our v3 DPO showed margin=30.436 and accuracy=100% — indicating very clean preference separation

**Tags:** #diamond-knowledge #monitoring #vram #reward-tracking

See also: [[Training Agent]] [[Evaluation Agent]]
