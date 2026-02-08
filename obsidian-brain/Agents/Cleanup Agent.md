# Cleanup Agent Memory

Persistent knowledge store for repository organization insights.

**Agent file:** `.github/agents/cleanup-agent.md`  
**Tags:** #agent/cleanup

---

## [2026-02-08] Major repository cleanup

**Context:** Comprehensive cleanup reduced repo from ~13GB to 235MB.  
**Insight:** Purged 2,452 invalid artifact files, 1,863 log files, 6 old training data directories, 8.4GB archive of unsuitable datasets, 4GB temp files. All pre-2026-02-08 pipeline data was invalid due to pipeline.py bug.  
**Application:** After any pipeline bug fix, always invalidate and clean ALL artifacts from before the fix date.  
**Tags:** #insight #cleanup

See also: [[References/Pipeline Bug 2026-02-08]]
