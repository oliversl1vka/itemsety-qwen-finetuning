"""LLM Council Advisor for fine-tuning evaluation and training review.

Implements a 3-stage multi-LLM council process inspired by
karpathy/llm-council (https://github.com/karpathy/llm-council),
using OpenRouter to provide multi-perspective analysis of:

  1. Fine-tuning evaluation results  (precision / recall / F1 / Jaccard)
  2. Training script configuration   (LoRA settings, hyperparams, data)

Council process
---------------
Stage 1 – Individual opinions : Each council LLM independently analyses the query.
Stage 2 – Peer review         : Each LLM reads anonymised responses and ranks them.
Stage 3 – Chairman synthesis  : A designated chairman LLM produces a final verdict.

Setup
-----
Add ``OPENROUTER_API_KEY`` to the environment *or* to ``openrouter.env``::

    OPENROUTER_API_KEY=sk-or-v1-...

Get a key at https://openrouter.ai/

Standalone CLI usage
--------------------
::

    # Analyse evaluation results
    python src/evaluation/council_advisor.py analyze \\
        --eval-results eval_results/evaluation_summary.json

    # Get training-script improvement advice
    python src/evaluation/council_advisor.py advise \\
        --training-script src/training/run_sft_full.py \\
        --eval-results eval_results/evaluation_summary.json

    # Override council models / chairman
    python src/evaluation/council_advisor.py analyze \\
        --eval-results eval_results/evaluation_summary.json \\
        --council-models anthropic/claude-sonnet-4.6 google/gemini-3-flash-preview \\
        --chairman anthropic/claude-opus-4.6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Default council configuration (affordable, capable models via OpenRouter)
# ---------------------------------------------------------------------------

DEFAULT_COUNCIL_MODELS: List[str] = [
    "anthropic/claude-sonnet-4.6",         # Frontier Sonnet – top-tier coding & analysis
    "google/gemini-3-flash-preview",       # Near-Pro reasoning at Flash speed
    "deepseek/deepseek-v3.2",             # GPT-5-class reasoning, very affordable
    "x-ai/grok-4.1-fast",                 # Strong agentic reasoning, 2M context
]

DEFAULT_CHAIRMAN_MODEL: str = "anthropic/claude-opus-4.6"  # Strongest synthesiser & analyst


# ---------------------------------------------------------------------------
# Import OpenRouter client
# ---------------------------------------------------------------------------

def _import_openrouter():
    """Import query_model / query_models_parallel from src.utils.openrouter_client."""
    # Ensure project root is on sys.path so the import always works regardless
    # of the working directory.
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from src.utils.openrouter_client import query_model, query_models_parallel
        return query_model, query_models_parallel
    except ImportError as exc:
        raise ImportError(
            "Could not import src.utils.openrouter_client.\n"
            "Make sure you are running from the project root and that "
            "src/utils/openrouter_client.py exists."
        ) from exc


# ---------------------------------------------------------------------------
# Stage 1 – Collect individual opinions
# ---------------------------------------------------------------------------

async def _stage1_collect_responses(
    query: str,
    council_models: List[str],
    query_model_fn,
    query_models_parallel_fn,
) -> List[Dict[str, Any]]:
    """Query every council model independently and collect their responses."""
    messages = [{"role": "user", "content": query}]
    print(f"  [Stage 1] Querying {len(council_models)} council models in parallel…")

    responses = await query_models_parallel_fn(council_models, messages)

    results = []
    for model, resp in responses.items():
        short = model.split("/")[-1]
        if resp is not None:
            results.append({"model": model, "response": resp.get("content", "")})
            print(f"    ✓ {short}")
        else:
            print(f"    ✗ {short} (failed – skipped)")

    return results


# ---------------------------------------------------------------------------
# Stage 2 – Peer rankings
# ---------------------------------------------------------------------------

async def _stage2_collect_rankings(
    query: str,
    stage1_results: List[Dict[str, Any]],
    query_model_fn,
    query_models_parallel_fn,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Have each council model anonymously rank the others' Stage-1 responses."""
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, …

    label_to_model: Dict[str, str] = {
        f"Response {lbl}": result["model"]
        for lbl, result in zip(labels, stage1_results)
    }

    responses_block = "\n\n".join(
        f"Response {lbl}:\n{result['response']}"
        for lbl, result in zip(labels, stage1_results)
    )

    ranking_prompt = (
        f"You are evaluating different expert analyses for the following query:\n\n"
        f"Query:\n{query}\n\n"
        f"Below are responses from different models (anonymised):\n\n"
        f"{responses_block}\n\n"
        "Your task:\n"
        "1. Evaluate each response individually — note what it does well and where it falls short.\n"
        "2. At the very end, provide a strict FINAL RANKING.\n\n"
        "IMPORTANT: The FINAL RANKING section MUST use this exact format:\n"
        "  FINAL RANKING:\n"
        "  1. Response <letter>\n"
        "  2. Response <letter>\n"
        "  …\n\n"
        "Example:\n"
        "  FINAL RANKING:\n"
        "  1. Response B\n"
        "  2. Response A\n"
        "  3. Response D\n"
        "  4. Response C\n\n"
        "Now provide your full evaluation followed by the FINAL RANKING section:"
    )

    messages = [{"role": "user", "content": ranking_prompt}]
    models_to_query = [r["model"] for r in stage1_results]

    print(f"  [Stage 2] Collecting peer rankings from {len(models_to_query)} models…")
    responses = await query_models_parallel_fn(models_to_query, messages)

    stage2_results = []
    for model, resp in responses.items():
        if resp is not None:
            full_text = resp.get("content", "")
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": _parse_ranking(full_text),
            })

    return stage2_results, label_to_model


# ---------------------------------------------------------------------------
# Stage 3 – Chairman synthesis
# ---------------------------------------------------------------------------

async def _stage3_synthesize_final(
    query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    chairman_model: str,
    query_model_fn,
) -> Dict[str, Any]:
    """Chairman LLM synthesises all Stage-1 + Stage-2 content into a final answer."""
    stage1_block = "\n\n".join(
        f"Model: {r['model']}\n{r['response']}" for r in stage1_results
    )
    stage2_block = "\n\n".join(
        f"Model: {r['model']}\nRanking response:\n{r['ranking']}" for r in stage2_results
    )

    chairman_prompt = (
        "You are the Chairman of an expert LLM Council. "
        "Multiple AI models have independently analysed the following query "
        "and then peer-reviewed each other's responses.\n\n"
        f"Original Query:\n{query}\n\n"
        "--- STAGE 1: Individual Analyses ---\n"
        f"{stage1_block}\n\n"
        "--- STAGE 2: Peer Rankings ---\n"
        f"{stage2_block}\n\n"
        "As Chairman, synthesise all insights into a single, comprehensive, actionable "
        "final response. Highlight areas of consensus, note important disagreements, "
        "and provide a clear, prioritised list of recommendations."
    )

    messages = [{"role": "user", "content": chairman_prompt}]
    chairman_short = chairman_model.split("/")[-1]
    print(f"  [Stage 3] Chairman ({chairman_short}) synthesising final response…")

    resp = await query_model_fn(chairman_model, messages, timeout=600.0)
    if resp is None:
        return {"model": chairman_model, "response": "⚠️ Chairman failed to produce a response."}

    return {"model": chairman_model, "response": resp.get("content", "")}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ranking(text: str) -> List[str]:
    """Extract the ordered list of 'Response X' labels from a FINAL RANKING block."""
    if "FINAL RANKING:" in text:
        _, _, section = text.partition("FINAL RANKING:")
        numbered = re.findall(r"\d+\.\s*Response [A-Z]", section)
        if numbered:
            return [re.search(r"Response [A-Z]", m).group() for m in numbered]
    # Fallback: grab any 'Response X' occurrence in order
    return re.findall(r"Response [A-Z]", text)


def _calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Compute average rank position for each model across all peer evaluations."""
    positions: Dict[str, List[int]] = defaultdict(list)

    for ranking in stage2_results:
        for pos, label in enumerate(ranking.get("parsed_ranking", []), start=1):
            if label in label_to_model:
                positions[label_to_model[label]].append(pos)

    aggregate = [
        {
            "model": model,
            "average_rank": round(sum(pos) / len(pos), 2),
            "rankings_count": len(pos),
        }
        for model, pos in positions.items()
        if pos
    ]
    aggregate.sort(key=lambda x: x["average_rank"])
    return aggregate


# ---------------------------------------------------------------------------
# Full council runner
# ---------------------------------------------------------------------------

async def _run_council(
    query: str,
    council_models: List[str],
    chairman_model: str,
) -> Dict[str, Any]:
    """Execute the full 3-stage council process and return structured results."""
    query_model_fn, query_models_parallel_fn = _import_openrouter()

    # Stage 1
    stage1 = await _stage1_collect_responses(
        query, council_models, query_model_fn, query_models_parallel_fn
    )
    if not stage1:
        return {"error": "All council models failed to respond in Stage 1."}

    # Stage 2
    stage2, label_to_model = await _stage2_collect_rankings(
        query, stage1, query_model_fn, query_models_parallel_fn
    )

    # Aggregate rankings
    aggregate = _calculate_aggregate_rankings(stage2, label_to_model)

    # Stage 3
    stage3 = await _stage3_synthesize_final(
        query, stage1, stage2, chairman_model, query_model_fn
    )

    return {
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "metadata": {
            "label_to_model": label_to_model,
            "aggregate_rankings": aggregate,
            "council_models": council_models,
            "chairman_model": chairman_model,
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_eval_analysis_query(summary: Dict[str, Any]) -> str:
    """Build the evaluation-results analysis query for the council."""
    cfg = summary.get("config", {})
    model_name = cfg.get("model", "Unknown model")
    comparison_rows = summary.get("comparison_rows", [])
    validation = summary.get("validation", {})
    best_run = summary.get("best_run", {})
    worst_run = summary.get("worst_run", {})

    lines = [
        "You are an expert in NLP model evaluation, frequent itemset mining, and LLM fine-tuning.",
        "",
        "A Qwen2.5 model has been fine-tuned with LoRA to extract frequent itemsets from CSV",
        "transactional data. The model is evaluated against the Apriori algorithm (ground truth).",
        "",
        f"Model adapter : {model_name}",
        f"Min support   : {cfg.get('min_support', 'N/A')}",
        f"Max itemset   : {cfg.get('max_size', 'N/A')}",
        "",
        "EVALUATION RESULTS",
        "------------------",
        f"  Datasets evaluated : {summary.get('successful_evaluations', '?')}"
            f" / {summary.get('total_datasets', '?')}",
        f"  Failed evaluations : {summary.get('failed_evaluations', 0)}",
        f"  Average Precision  : {summary.get('avg_precision', 0):.2%}",
        f"  Average Recall     : {summary.get('avg_recall', 0):.2%}",
        f"  Average F1 Score   : {summary.get('avg_f1_score', 0):.2%}",
        f"  Average Jaccard    : {summary.get('avg_jaccard', 0):.2%}",
        f"  Exact Match Rate   : {summary.get('exact_match_rate', 0):.2%}",
        f"  Exact Matches      : {summary.get('exact_match_count', 0)}"
            f" / {summary.get('successful_evaluations', '?')}",
        "",
        "TARGETS",
        "-------",
        "  F1 Score          ≥ 0.80   (primary target)",
        "  Exact Match Rate  ≥ 0.50",
        "  JSON Parse Rate   ≥ 0.90",
        "",
        "TRAINING CONTEXT",
        "----------------",
        "  Task           : Extract ALL frequent itemsets (co-occurring ≥ min_support times).",
        "  Training data  : ~440 ChatML examples generated by GPT-4.1 from 500 CSV datasets.",
        "  Base model     : Qwen2.5-7B-Instruct.",
        "  Fine-tuning    : LoRA (r=16, alpha=32), 3 epochs, LR=2e-4, effective batch=16.",
        "  Output format  : JSON array [{\"itemset\": [...], \"support\": N, \"rows\": [...]}].",
        "",
    ]

    if validation:
        lines += [
            "EVALUATION VALIDATION CONTEXT",
            "-----------------------------",
            f"  Eval dir validated      : {validation.get('eval_dir_validated', 'N/A')}",
            f"  Expected eval prefix    : {validation.get('expected_eval_prefix', 'N/A')}",
            f"  Prefix validation OK    : {validation.get('prefix_ok', 'N/A')}",
            f"  Training overlap count  : {validation.get('training_overlap_count', 'N/A')}",
            f"  Saved artifact files    : {validation.get('saved_files', 'N/A')}",
            "",
        ]

    if comparison_rows:
        lines += [
            "MULTI-RUN COMPARISON",
            "--------------------",
            "These results compare multiple adapters and decoding profiles from the same unseen eval set.",
            "Use this table to distinguish decoding effects from genuine model-quality changes.",
            "",
            "model_key | model_label | profile_key | avg_precision | avg_recall | avg_f1 | hit_limit_rate | duplicate_itemset_rate | avg_output_tokens | avg_time_s",
        ]
        for row in comparison_rows:
            lines.append(
                "  {model_key} | {model_label} | {profile_key} | {avg_precision:.2%} | {avg_recall:.2%} | {avg_f1:.2%} | {hit_limit_rate:.2%} | {duplicate_itemset_rate:.2%} | {avg_output_tokens:.1f} | {avg_time_s:.1f}".format(
                    model_key=row.get("model_key", "N/A"),
                    model_label=row.get("model_label", "N/A"),
                    profile_key=row.get("profile_key", "N/A"),
                    avg_precision=row.get("avg_precision", 0.0),
                    avg_recall=row.get("avg_recall", 0.0),
                    avg_f1=row.get("avg_f1", 0.0),
                    hit_limit_rate=row.get("hit_limit_rate", 0.0),
                    duplicate_itemset_rate=row.get("duplicate_itemset_rate", 0.0),
                    avg_output_tokens=row.get("avg_output_tokens", 0.0),
                    avg_time_s=row.get("avg_time_s", 0.0),
                )
            )
        lines.append("")

    if best_run:
        lines += [
            "BEST OBSERVED RUN",
            "-----------------",
            f"  Model/Profile       : {best_run.get('model_label', 'N/A')} / {best_run.get('profile_key', 'N/A')}",
            f"  Avg F1              : {best_run.get('avg_f1', 0):.2%}",
            f"  Avg Precision       : {best_run.get('avg_precision', 0):.2%}",
            f"  Avg Recall          : {best_run.get('avg_recall', 0):.2%}",
            f"  Hit limit rate      : {best_run.get('hit_limit_rate', 0):.2%}",
            f"  Duplicate rate      : {best_run.get('duplicate_itemset_rate', 0):.2%}",
            "",
        ]

    if worst_run:
        lines += [
            "WORST OBSERVED RUN",
            "------------------",
            f"  Model/Profile       : {worst_run.get('model_label', 'N/A')} / {worst_run.get('profile_key', 'N/A')}",
            f"  Avg F1              : {worst_run.get('avg_f1', 0):.2%}",
            f"  Avg Precision       : {worst_run.get('avg_precision', 0):.2%}",
            f"  Avg Recall          : {worst_run.get('avg_recall', 0):.2%}",
            f"  Hit limit rate      : {worst_run.get('hit_limit_rate', 0):.2%}",
            f"  Duplicate rate      : {worst_run.get('duplicate_itemset_rate', 0):.2%}",
            "",
        ]

    lines += [
        "Please provide a thorough analysis covering:",
        "  1. Overall performance assessment (good / acceptable / poor and why).",
        "  2. Key strengths and weaknesses visible in the metrics.",
        "  3. Likely root causes for any performance gaps.",
        "  4. Specific, prioritised recommendations to improve the model.",
        "  5. Whether the fine-tuning approach is the right strategy vs. pure prompting.",
        "  6. What the multi-run comparison suggests about decoding settings vs. actual model quality.",
    ]
    return "\n".join(lines)


def _build_training_advice_query(
    script_content: str,
    eval_metrics: Optional[Dict[str, Any]],
) -> str:
    """Build the training-script advice query for the council."""
    lines = [
        "You are an expert in LLM fine-tuning, specifically LoRA/QLoRA training "
        "with HuggingFace PEFT and TRL.",
        "",
        "A Qwen2.5-7B model is being fine-tuned with LoRA to extract frequent itemsets "
        "from CSV transactional data. The task requires structured JSON output.",
        "",
    ]

    if eval_metrics:
        lines += [
            "CURRENT EVALUATION PERFORMANCE",
            "-------------------------------",
            f"  Average F1        : {eval_metrics.get('avg_f1_score', 0):.2%}",
            f"  Average Precision : {eval_metrics.get('avg_precision', 0):.2%}",
            f"  Average Recall    : {eval_metrics.get('avg_recall', 0):.2%}",
            f"  Exact Match Rate  : {eval_metrics.get('exact_match_rate', 0):.2%}",
            "",
            "TARGET: F1 ≥ 0.80",
            "",
        ]

    # Truncate very long scripts to stay within context limits
    max_chars = 6_000
    if len(script_content) > max_chars:
        script_content = (
            script_content[:max_chars]
            + "\n\n# … [script truncated for length] …"
        )

    lines += [
        "TRAINING SCRIPT",
        "---------------",
        "```python",
        script_content,
        "```",
        "",
        "Please review this training configuration and provide:",
        "  1. Assessment of the current hyperparameter choices.",
        "  2. Specific hyperparameter adjustments (with concrete values) to improve F1.",
        "  3. Data quality / quantity improvements worth considering.",
        "  4. Training stability improvements (gradient clipping, LR schedule, etc.).",
        "  5. Any architectural or methodological alternatives worth trying.",
        "  6. A prioritised action list (most impactful changes first).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API – CouncilAdvisor class
# ---------------------------------------------------------------------------

@dataclass
class CouncilAdvisor:
    """LLM Council Advisor: multi-LLM analysis for fine-tuning workflows.

    Example::

        from src.evaluation.council_advisor import CouncilAdvisor
        import json

        advisor = CouncilAdvisor()

        # Analyse evaluation results
        with open("eval_results/evaluation_summary.json") as f:
            summary = json.load(f)
        advisor.analyze_eval_results(summary, output_dir=Path("eval_results"))

        # Get training-script advice
        advisor.advise_training_script(
            script_path="src/training/run_sft_full.py",
            eval_metrics=summary,
            output_dir=Path("eval_results"),
        )
    """

    council_models: List[str] = field(
        default_factory=lambda: list(DEFAULT_COUNCIL_MODELS)
    )
    chairman_model: str = DEFAULT_CHAIRMAN_MODEL

    # ------------------------------------------------------------------
    def analyze_eval_results(
        self,
        summary: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Run a 3-stage council analysis on evaluation summary metrics.

        Args:
            summary:    Dict loaded from ``evaluation_summary.json``.
            output_dir: Directory where ``council_eval_analysis.json`` is saved.

        Returns:
            Full council results dict (stages 1–3 + metadata).
        """
        print("\n🔮 LLM Council: Analysing evaluation results…")
        query = _build_eval_analysis_query(summary)
        results = asyncio.run(
            _run_council(query, self.council_models, self.chairman_model)
        )

        if "error" in results:
            print(f"❌ Council failed: {results['error']}")
            return results

        self._save_and_print(
            results=results,
            output_path=Path(output_dir) / "council_eval_analysis.json",
            extra_fields={"query_type": "eval_analysis", "query": query},
            title="Council Evaluation Analysis",
        )
        return results

    # ------------------------------------------------------------------
    def advise_training_script(
        self,
        script_path: str,
        eval_metrics: Optional[Dict[str, Any]],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Run a 3-stage council review of a training script.

        Args:
            script_path:  Path to the training Python script.
            eval_metrics: Optional eval summary dict for additional context.
            output_dir:   Directory where ``council_training_advice.json`` is saved.

        Returns:
            Full council results dict (stages 1–3 + metadata).
        """
        print(f"\n🔧 LLM Council: Reviewing training script ({script_path})…")

        try:
            script_content = Path(script_path).read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"❌ Training script not found: {script_path}")
            return {"error": f"Script not found: {script_path}"}

        query = _build_training_advice_query(script_content, eval_metrics)
        results = asyncio.run(
            _run_council(query, self.council_models, self.chairman_model)
        )

        if "error" in results:
            print(f"❌ Council failed: {results['error']}")
            return results

        self._save_and_print(
            results=results,
            output_path=Path(output_dir) / "council_training_advice.json",
            extra_fields={
                "query_type": "training_advice",
                "script_path": script_path,
                "query": query,
            },
            title="Council Training Advice",
        )
        return results

    # ------------------------------------------------------------------
    def _save_and_print(
        self,
        results: Dict[str, Any],
        output_path: Path,
        extra_fields: Dict[str, Any],
        title: str,
    ) -> None:
        """Save results to JSON and print the chairman's final response."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({**extra_fields, **results}, f, indent=2, ensure_ascii=False)

        # Print chairman synthesis
        chairman_response = results.get("stage3", {}).get("response", "")
        sep = "=" * 60
        print(f"\n🏛️  {title}:\n{sep}")
        print(chairman_response)
        print(sep)
        print(f"\n💾 Full council results → {output_path}")

        # Print aggregate rankings
        agg = results.get("metadata", {}).get("aggregate_rankings", [])
        if agg:
            print("\n📊 Response Rankings (avg rank ↓ = better):")
            for item in agg:
                short = item["model"].split("/")[-1]
                print(f"   {item['average_rank']:.2f}  {short}  ({item['rankings_count']} votes)")


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="council_advisor",
        description=(
            "LLM Council Advisor: multi-LLM analysis for fine-tuning evaluation "
            "and training script review."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              # Analyse evaluation results
              python src/evaluation/council_advisor.py analyze \\
                  --eval-results eval_results/evaluation_summary.json

              # Get training-script improvement advice
              python src/evaluation/council_advisor.py advise \\
                  --training-script src/training/run_sft_full.py \\
                  --eval-results eval_results/evaluation_summary.json

              # Override models
              python src/evaluation/council_advisor.py analyze \\
                  --eval-results eval_results/evaluation_summary.json \\
                  --council-models anthropic/claude-sonnet-4.6 google/gemini-3-flash-preview \\
                  --chairman anthropic/claude-opus-4.6
            """
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- analyze ---
    analyze_p = subparsers.add_parser(
        "analyze", help="Analyse evaluation results with the LLM council."
    )
    analyze_p.add_argument(
        "--eval-results",
        required=True,
        metavar="PATH",
        help="Path to evaluation_summary.json produced by eval_finetuned_model.py.",
    )
    analyze_p.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Output directory (defaults to the same folder as --eval-results).",
    )

    # --- advise ---
    advise_p = subparsers.add_parser(
        "advise", help="Get council advice on a training script."
    )
    advise_p.add_argument(
        "--training-script",
        default="src/training/run_sft_full.py",
        metavar="PATH",
        help="Path to the training Python script (default: src/training/run_sft_full.py).",
    )
    advise_p.add_argument(
        "--eval-results",
        default=None,
        metavar="PATH",
        help="Path to evaluation_summary.json for extra context (optional).",
    )
    advise_p.add_argument(
        "--output-dir",
        default="eval_results",
        metavar="DIR",
        help="Output directory for council results (default: eval_results).",
    )

    # Shared model options
    for sub in (analyze_p, advise_p):
        sub.add_argument(
            "--council-models",
            nargs="+",
            default=None,
            metavar="MODEL",
            help="Override council model list (space-separated OpenRouter identifiers).",
        )
        sub.add_argument(
            "--chairman",
            default=None,
            metavar="MODEL",
            help="Override chairman model (OpenRouter identifier).",
        )

    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()

    # Build advisor with optional overrides
    advisor_kwargs: Dict[str, Any] = {}
    if args.council_models:
        advisor_kwargs["council_models"] = args.council_models
    if args.chairman:
        advisor_kwargs["chairman_model"] = args.chairman

    advisor = CouncilAdvisor(**advisor_kwargs)

    # ---- analyze ----
    if args.command == "analyze":
        eval_path = Path(args.eval_results)
        if not eval_path.exists():
            print(f"❌ File not found: {eval_path}")
            sys.exit(1)

        with open(eval_path, encoding="utf-8") as f:
            summary = json.load(f)

        out_dir = Path(args.output_dir) if args.output_dir else eval_path.parent
        results = advisor.analyze_eval_results(summary, out_dir)
        sys.exit(0 if "error" not in results else 1)

    # ---- advise ----
    elif args.command == "advise":
        eval_metrics = None
        if args.eval_results:
            eval_path = Path(args.eval_results)
            if eval_path.exists():
                with open(eval_path, encoding="utf-8") as f:
                    eval_metrics = json.load(f)
            else:
                print(f"⚠️  --eval-results not found ({args.eval_results}), proceeding without metrics.")

        results = advisor.advise_training_script(
            script_path=args.training_script,
            eval_metrics=eval_metrics,
            output_dir=Path(args.output_dir),
        )
        sys.exit(0 if "error" not in results else 1)


if __name__ == "__main__":
    main()
