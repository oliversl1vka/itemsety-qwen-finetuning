"""Async OpenRouter API client.

Provides parallel model querying for multi-LLM workflows.
Inspired by karpathy/llm-council (https://github.com/karpathy/llm-council).

Usage:
    from src.utils.openrouter_client import query_model, query_models_parallel

    # Single model (sync wrapper)
    import asyncio
    response = asyncio.run(query_model("openai/gpt-4o-mini", messages))

    # Multiple models in parallel
    responses = asyncio.run(query_models_parallel(models, messages))

Setup:
    Add OPENROUTER_API_KEY to environment or to openrouter.env:
        OPENROUTER_API_KEY=sk-or-v1-...

    Get a key at https://openrouter.ai/
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_HTTP_REFERER = "https://github.com/oliversl1vka/itemsety-qwen-finetuning"
_APP_TITLE = "Itemsety Qwen Fine-tuning Advisor"


# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------

def _load_api_key() -> Optional[str]:
    """Load OPENROUTER_API_KEY from environment or local .env files.

    Search order:
      1. OPENROUTER_API_KEY environment variable
      2. openrouter.env  (project root)
      3. .env            (project root)
    """
    # 1. Environment variable (highest priority)
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key

    # 2. & 3. Local env files
    search_dirs = [
        Path(__file__).resolve().parent.parent.parent,  # project root
        Path.cwd(),
    ]
    for base in search_dirs:
        for filename in ("openrouter.env", ".env"):
            env_path = base / filename
            if env_path.exists():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    if k.strip() == "OPENROUTER_API_KEY":
                        val = v.strip().strip("\"'")
                        if val:
                            return val

    return None


# ---------------------------------------------------------------------------
# Core async API functions
# ---------------------------------------------------------------------------

async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 600.0,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """Query a single model via the OpenRouter API with retry logic.

    Args:
        model:       OpenRouter model identifier, e.g. ``"openai/gpt-4o-mini"``.
        messages:    Chat message list ``[{"role": ..., "content": ...}]``.
        timeout:     HTTP request timeout in seconds (default 600s for long queries).
        max_retries: Number of retry attempts on transient failures (default 3).

    Returns:
        Dict with ``"content"`` key (str), or ``None`` if all attempts fail.

    Raises:
        ValueError: If ``OPENROUTER_API_KEY`` is not found anywhere.
        ImportError: If ``httpx`` is not installed.
    """
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required for OpenRouter queries. Install it with:\n"
            "  pip install httpx"
        ) from exc

    api_key = _load_api_key()
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found.\n"
            "Set it as an environment variable or add it to openrouter.env:\n"
            "  OPENROUTER_API_KEY=sk-or-v1-..."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": _HTTP_REFERER,
        "X-Title": _APP_TITLE,
    }

    payload = {"model": model, "messages": messages}
    model_short = model.split("/")[-1]

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                message = data["choices"][0]["message"]
                return {"content": message.get("content", "")}
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = 2 ** attempt  # 2s, 4s, 8s exponential backoff
                print(f"  ⚠️  [{model_short}] Attempt {attempt}/{max_retries} failed: {exc}")
                print(f"      Retrying in {wait}s…")
                await asyncio.sleep(wait)
            else:
                print(f"  ⚠️  [{model_short}] All {max_retries} attempts failed: {last_exc}")
                return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Query multiple models in parallel via ``asyncio.gather()``.

    Args:
        models:   List of OpenRouter model identifiers.
        messages: Chat messages to send to every model.

    Returns:
        Dict mapping model identifier → response dict (or ``None`` on failure).
    """
    tasks = [query_model(model, messages) for model in models]
    responses = await asyncio.gather(*tasks)
    return dict(zip(models, responses))
