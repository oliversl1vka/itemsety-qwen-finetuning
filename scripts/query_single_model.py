"""Query a single model and append to council_v3_partial.json."""

import httpx
import json
import sys
import time
from pathlib import Path

ROOT = Path("/Users/oliver/itemsety-qwen-finetuning")
PARTIAL = ROOT / "docs" / "reports" / "council_v3_partial.json"
URL = "https://openrouter.ai/api/v1/chat/completions"


def load_key():
    for line in (ROOT / "openrouter.env").read_text().splitlines():
        if "OPENROUTER_API_KEY" in line and "=" in line and not line.startswith("#"):
            return line.partition("=")[2].strip().strip("\"'")
    raise ValueError("No key found")


def main():
    model = sys.argv[1]
    api_key = load_key()
    model_short = model.split("/")[-1]

    # Load partial results
    results = json.loads(PARTIAL.read_text())
    existing = [r["model"] for r in results["stage1"]]
    if model in existing:
        print(f"⚠️  {model_short} already in results, skipping")
        return

    # Build query
    sys.path.insert(0, str(ROOT))
    from scripts.council_v3_sequential import build_query
    query = build_query()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/oliversl1vka/itemsety-qwen-finetuning",
    }
    payload = {"model": model, "messages": [{"role": "user", "content": query}]}

    for attempt in range(1, 4):
        print(f"[{model_short}] Attempt {attempt}/3...")
        t0 = time.time()
        try:
            with httpx.Client(timeout=600.0) as client:
                resp = client.post(URL, headers=headers, json=payload)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"].get("content", "")
                elapsed = time.time() - t0
                print(f"✅ {model_short}: {elapsed:.1f}s, {len(content)} chars")

                results["stage1"].append({"model": model, "response": content})
                PARTIAL.write_text(json.dumps(results, indent=2, ensure_ascii=False))
                total = len([r for r in results["stage1"] if r.get("response")])
                print(f"💾 Saved ({total} responses total)")
                return
        except Exception as e:
            elapsed = time.time() - t0
            print(f"❌ {model_short} attempt {attempt} failed ({elapsed:.1f}s): {e}")
            if attempt < 3:
                wait = 5 * attempt
                print(f"   Waiting {wait}s...")
                time.sleep(wait)

    print(f"❌ {model_short} FAILED all attempts")
    results["stage1"].append({"model": model, "response": None, "failed": True})
    PARTIAL.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
