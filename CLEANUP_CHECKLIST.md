# Pre-Documentation Cleanup Checklist

> **Philosophy:** Docs are written LAST, in one shot, on a clean repo.  
> Complete every item in this checklist before opening `DOCS_PLAN.md`.  
> Check off items as you go. When this list is fully checked, start writing docs.

---

## CRITICAL — Must complete before going public

- [ ] **Rotate your OpenAI API key** — go to platform.openai.com → API keys → revoke the current key → create a new one → update your local `openai.env`. The old key was loaded in a Claude Code session and should be treated as exposed.
- [ ] **Rotate your HuggingFace token** — huggingface.co → Settings → Access Tokens → revoke → create new → update `hf.env`
- [ ] **Rotate your OpenRouter key** — openrouter.ai → Keys → revoke → create new → update `openrouter.env`
- [ ] **Verify no secrets in git history:**
  ```bash
  git log --all --full-history --oneline -- '**/openai.env' '**/hf.env' '**/openrouter.env' '**/*.env'
  git grep -n 'sk-proj-\|sk-or-v1-\|hf_' $(git log --format='%H')
  ```
  If anything shows up — stop and do a history rewrite before proceeding.

---

## Untracked Files (resolve each one)

Run `git status` — you should see these untracked files. Decide for each:

- [ ] `.claude/` — **gitignore** (Claude Code metadata, already handled)
- [ ] `2026-04-25-142700-this-session-is-being-continued-from-a-previous-c.txt` — **delete or gitignore** (conversation export, no value in public repo)
- [ ] `eval_datasets_v2_2026-04-12.zip` — **gitignore** (already added to .gitignore)
- [ ] `eval_results/` — **DECISION NEEDED:**
  - Does this directory contain the canonical evaluation results for your thesis?
  - If **yes** (these are the numbers you'll cite): commit it with a `eval_results/README.md` explaining which model checkpoint and eval set produced these numbers
  - If **no** (intermediate/scratch results): gitignore it
- [ ] `notebooks/eval_model_2026-04-12_v3.ipynb` — **DECISION NEEDED:**
  - Is this your canonical evaluation notebook (the one that produced final thesis numbers)?
  - If **yes**: clear outputs (`jupyter nbconvert --clear-output`), commit, move non-canonical ones to `notebooks/archive/`
  - If **no**: gitignore or archive

---

## Modified Files (per git status — review and commit)

These files have uncommitted changes. Review each, understand what changed, then commit:

- [ ] `.github/agents/training-agent.md` — review what changed, commit
- [ ] `.github/copilot-instructions.md` — review for any sensitive internal content before committing; this will be publicly visible
- [ ] `notebooks/training_3phase_2026-03-09_v3.ipynb` — **important:** clear all cell outputs first (outputs may contain API responses or personal paths), then commit
- [ ] `obsidian-brain/Agents/Training Agent.md` — review and commit
- [ ] `src/evaluation/council_advisor.py` — review changes, commit

After reviewing all five, do a single clean commit:
```bash
git add .github/agents/training-agent.md .github/copilot-instructions.md \
  notebooks/training_3phase_2026-03-09_v3.ipynb \
  "obsidian-brain/Agents/Training Agent.md" \
  src/evaluation/council_advisor.py
git commit -m "chore: review and commit modified files pre-publication"
```

---

## Notebooks

- [ ] **Decide on the canonical training notebook** — likely `training_3phase_2026-03-09_v3.ipynb`. Rename it or symlink to `notebooks/training_canonical.ipynb` so the docs can reference a stable filename. Update `DOCS_PLAN.md` references if you rename.
- [ ] **Archive non-canonical notebooks** — move these to `notebooks/archive/`:
  - `training_sft_dpo_template.ipynb` (legacy 2-phase template)
  - `eval_model.ipynb` (legacy eval)
  - `eval_v2_model.ipynb` (v2 eval)
  - `training_3phase_2026-03-07_v2.ipynb` (older version)
  - Keep: `training_3phase_7b.ipynb` and/or `training_3phase_2026-03-09_v3.ipynb`
- [ ] **Clear all notebook outputs** from any notebook you commit:
  ```bash
  jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
  ```
  Outputs can contain API keys in error messages, paths, or raw model responses.
- [ ] **Verify Cell 2 of canonical training notebook** — confirm all hyperparameters match what's in `DOCS_PLAN.md` (Section 5, ADR-013 and ADR-014). These are the numbers the docs will cite.

---

## Evaluation Results — Collect the Numbers

The docs have `?` placeholders for all eval metrics. Before writing docs, you need:

- [ ] Run `src/evaluation/eval_finetuned_model.py` against your final model checkpoint if you haven't already
- [ ] Collect these numbers (for README + docs/index.md + docs/methodology/evaluation.md):

  | Model | P | R | F1 | Exact Match | Hallucination Rate | JSON Parse |
  |---|---|---|---|---|---|---|
  | Base Qwen2.5-7B | | | | | | |
  | + SFT (phase 1) | | | | | | |
  | + SFT+DPO (final) | | | | | | |
  | GPT-4.1-mini baseline | | | | | | |

- [ ] Note which model checkpoint produced the final numbers (HF Hub path)
- [ ] Note which eval set version was used (eval_datasets_v2?)

---

## HuggingFace Hub

- [ ] **Verify model is pushed and loadable:**
  ```python
  from unsloth import FastLanguageModel
  model, tokenizer = FastLanguageModel.from_pretrained("OliverSlivka/qwen2.5-7b-itemset-extractor")
  ```
- [ ] **Verify dataset is accessible:** `OliverSlivka/itemset-extraction-v3` — check on huggingface.co
- [ ] **Write a Model Card** on HuggingFace for `OliverSlivka/qwen2.5-7b-itemset-extractor` — minimum: what it does, how to use it, training details, link to this repo. HF renders the README.md of the model repo as the Model Card.
- [ ] **Verify the Gradio demo** (if deployed to HF Spaces) — test it loads and runs correctly. Get the URL for the docs.
- [ ] (Optional) **Push eval_datasets_v2 to HF** as a separate dataset for full reproducibility

---

## Obsidian Brain — Option B

- [ ] **Add `obsidian-brain/README.md`** explaining what the vault is:
  ```markdown
  # Obsidian Brain — AI Agent Memory Vault
  
  This directory is the persistent memory vault for the 9 AI agents that developed this project.
  It contains architectural decision records, experiment logs, reference notes, and agent memory files.
  
  The key content is mirrored and polished in the main documentation site:
  - Decisions → docs/decisions/ (ADRs)
  - Experiments → docs/ai-workflow/experiment-journal.md
  - Agent system → docs/ai-workflow/agent-system.md
  
  This vault is preserved as an audit trail of the AI-augmented development process.
  ```
- [ ] **Do not modify the vault content** — it's the authentic audit trail. Only add the README.

---

## Existing `docs/` Directory

The current `docs/` directory has guides, reports, and archived content. Decide what to do with it before MkDocs takes over the `docs/` folder:

- [ ] **Move existing `docs/` to `docs/legacy/`** to preserve it without conflict with new MkDocs content:
  ```bash
  mkdir -p docs/legacy
  mv docs/guides docs/legacy/
  mv docs/reports docs/legacy/
  mv docs/archive docs/legacy/
  mv docs/thesis_structure.md docs/legacy/
  # leave docs/data/ — it's a data README, keep it
  ```
  Then add `docs/legacy/` to the MkDocs nav under a "Legacy Docs" section (optional — or just leave it on disk without a nav entry, MkDocs will ignore unlisted files)

- [ ] **Review `docs/reports/`** — do any report numbers contradict your final eval results? If yes, annotate those reports with a note pointing to the canonical results. Don't delete them — they're the research history.

- [ ] **`docs/thesis_structure.md`** — is this still accurate? Update or move to legacy.

---

## Code Quality

- [ ] **Check `pipeline.py` for TODO/FIXME/debug prints** that shouldn't be public:
  ```bash
  grep -n 'TODO\|FIXME\|HACK\|print(' pipeline.py | head -30
  ```
- [ ] **Check `src/` for the same:**
  ```bash
  grep -rn 'TODO\|FIXME\|HACK' src/
  ```
  These are fine if they're genuine technical notes; only remove if embarrassing or misleading.
- [ ] **Verify `requirements.txt` is complete** — does a fresh `pip install -r requirements.txt` work in a clean environment? (test on the server or locally)

---

## GitHub Repository Settings (do in GitHub UI)

Do these after pushing the cleaned repo and before making it public:

- [ ] **Repository description:** "Fine-tuning Qwen2.5-7B for frequent itemset extraction · Apriori oracle · SFT+DPO · Bachelor's Thesis 2026 · FIS VŠE Prague"
- [ ] **Topics/tags:** `machine-learning` `nlp` `qwen` `fine-tuning` `lora` `dpo` `frequent-itemsets` `bachelor-thesis` `unsloth` `transformers` `frequent-itemset-mining` `itemset-extraction`
- [ ] **Website URL:** set to MkDocs Pages URL once deployed (will be `https://YOUR_USERNAME.github.io/itemsety-qwen-finetuning/`)
- [ ] **Social preview image:** upload a clean diagram of the pipeline (optional but professional)
- [ ] **Enable GitHub Pages:** Settings → Pages → Source → GitHub Actions (needed for the docs workflow)
- [ ] **Make repository public:** Settings → Danger Zone → Change visibility → Public (do this LAST)

---

## Final Verification Before Writing Docs

Run through this checklist as the final gate:

- [ ] `git status` shows a clean working tree (no untracked or modified files)
- [ ] `git ls-files | grep '\.env'` — returns nothing (no env files tracked)
- [ ] The repo makes sense as a public visitor's first impression
- [ ] You have all the evaluation numbers written down (the `?` placeholders in DOCS_PLAN.md are filled)
- [ ] The canonical training notebook is identified and committed
- [ ] The canonical eval notebook is identified and committed  
- [ ] The HF model is live and loadable
- [ ] A GitHub Release hasn't been tagged yet (do this after docs are done)

---

## After Docs Are Done

- [ ] Tag the release: `git tag -a v1.0.0 -m "Bachelor's thesis submission" && git push origin v1.0.0`
- [ ] Create a GitHub Release from the tag with release notes
- [ ] Register on [Zenodo](https://zenodo.org) → link your GitHub → the v1.0.0 release gets a DOI automatically
- [ ] Add the Zenodo DOI badge to the README and `CITATION.cff`
- [ ] Submit the repo link and DOI to your thesis
