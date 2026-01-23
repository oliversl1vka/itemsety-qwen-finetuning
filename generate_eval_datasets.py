"""Generate evaluation datasets for fine-tuned model evaluation.

Creates 50 new synthetic datasets in eval_datasets/ directory for comparing
fine-tuned Qwen model against ground-truth Apriori results.

These datasets are SEPARATE from training data to avoid data leakage.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import string
import sys
from datetime import datetime, UTC
from pathlib import Path
from secrets import token_hex
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================
# EVAL CONFIGURATION
# =========================

EVAL_CONFIG = {
    # Generate 50 evaluation datasets
    "num_datasets": 50,
    
    # Slightly different size ranges to test generalization
    "rows_range": (8, 60),  # Smaller range for faster evaluation
    "cols_range": (15, 80),
    
    # Column type probabilities (same as training)
    "col_type_probs": {"int": 0.35, "float": 0.25, "text": 0.30, "bool": 0.10},
    
    # Numeric ranges
    "int_range": (-10_000, 10_000),
    "float_range": (-1e6, 1e6),
    "float_decimals": 4,
    
    # Text tokens
    "num_synthetic_words": 50_000,
    "curated_tokens": {
        "colors": ["red", "green", "blue", "yellow", "black", "white", "orange", "purple", "pink", "brown", "gray"],
        "animals": ["ant", "bee", "cat", "dog", "fox", "owl", "yak", "zebra"],
        "products": ["phone", "laptop", "mug", "bottle", "chair", "lamp", "watch", "backpack"],
        "cities": ["prague", "vienna", "berlin", "paris", "london", "madrid", "rome"],
    },
    "frequent_token_bias": 0.15,
    
    # Hash perturbations
    "enable_hash_perturb": True,
    "hash_iterations": (1, 3),
    "hash_algos": ["md5", "sha1", "sha256", "blake2b", "blake2s"],
    "perturb_ops": ["shuffle_rows", "permute_cols", "inject_rare_tokens", "noise_numeric", "bucketize_some"],
    
    # Output settings - SEPARATE directory for evaluation
    "output_dir": "eval_datasets",
    "prefix": "eval",
    "log_filename": "logs/eval_generation_log.csv",
}


# =========================
# Utility functions
# =========================

def hash_to_seed(s: str, algo: str = "sha256") -> int:
    h = getattr(hashlib, algo)(s.encode()).hexdigest()
    return int(h[:16], 16)


def random_syllable() -> str:
    return random.choice("bcdfghjklmnpqrstvwxyz") + random.choice("aeiou")


def make_synthetic_word(min_syl: int = 2, max_syl: int = 4) -> str:
    return "".join(random_syllable() for _ in range(random.randint(min_syl, max_syl)))


def build_text_lexicon(num_synth: int, curated: Dict[str, List[str]]) -> List[str]:
    base: List[str] = []
    for v in curated.values():
        base.extend(v)
    base.extend({make_synthetic_word() for _ in range(num_synth)})
    random.shuffle(base)
    return list(set(base))


# =========================
# Column generators
# =========================

def gen_int_column(n: int, r: Tuple[int, int]) -> List[int]:
    return [random.randint(*r) for _ in range(n)]


def gen_float_column(n: int, r: Tuple[float, float], dec: int) -> List[float]:
    return [round(random.uniform(*r), dec) for _ in range(n)]


def gen_bool_column(n: int) -> List[str]:
    return [random.choice(["True", "False"]) for _ in range(n)]


def gen_text_column(n: int, lexicon: List[str], bias: float = 0.0) -> List[str]:
    k = max(3, int(0.01 * len(lexicon)))
    dominants = random.sample(lexicon, k)
    p_dom = bias
    p_rest = 1 - p_dom
    weights_dom = [p_dom / len(dominants)] * len(dominants)
    weights_rest = [p_rest / (len(lexicon) - len(dominants))] * (len(lexicon) - len(dominants))
    choices = dominants + [w for w in lexicon if w not in dominants]
    weights = weights_dom + weights_rest
    return random.choices(choices, weights=weights, k=n)


def draw_col_types(n: int, probs: Dict[str, float]) -> List[str]:
    t = list(probs.keys())
    p = np.array(list(probs.values()))
    p /= p.sum()
    return list(np.random.choice(t, size=n, p=p))


# =========================
# Dataset creation
# =========================

def generate_base_dataset(rows: int, cols: int, cfg: Dict) -> pd.DataFrame:
    col_types = draw_col_types(cols, cfg["col_type_probs"])
    lexicon = build_text_lexicon(cfg["num_synthetic_words"], cfg["curated_tokens"])
    data: Dict[str, List] = {}
    for j, t in enumerate(col_types, 1):
        col = f"attr_{j}_{t}"
        if t == "int":
            data[col] = gen_int_column(rows, cfg["int_range"])
        elif t == "float":
            data[col] = gen_float_column(rows, cfg["float_range"], cfg["float_decimals"])
        elif t == "text":
            data[col] = gen_text_column(rows, lexicon, cfg["frequent_token_bias"])
        else:
            data[col] = gen_bool_column(rows)
    return pd.DataFrame(data)


# =========================
# Perturbations
# =========================

def perturb_shuffle_rows(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    return df.sample(frac=1, random_state=rng.randint(0, int(1e9))).reset_index(drop=True)


def perturb_permute_cols(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    cols = list(df.columns)
    rng.shuffle(cols)
    return df[cols]


def perturb_inject_rare_tokens(df: pd.DataFrame, rng: random.Random, strength: float = 0.02) -> pd.DataFrame:
    text_cols = [c for c in df.columns if "_text" in c]
    for c in text_cols:
        mask = np.random.rand(len(df)) < strength
        df.loc[mask, c] = "#" + "".join(rng.choice(string.hexdigits) for _ in range(12))
    return df


def perturb_noise_numeric(df: pd.DataFrame, rng: random.Random, noise_level: float = 0.05) -> pd.DataFrame:
    for c in [c for c in df.columns if "_int" in c or "_float" in c]:
        if "_int" in c:
            span = int(abs(EVAL_CONFIG["int_range"][1] - EVAL_CONFIG["int_range"][0]) * noise_level)
            noise = np.random.randint(-span, span + 1, len(df))
            df[c] = df[c].astype(int) + noise
        else:
            span = abs(EVAL_CONFIG["float_range"][1] - EVAL_CONFIG["float_range"][0]) * noise_level
            noise = np.random.uniform(-span, span, len(df))
            df[c] = df[c].astype(float) + noise
    return df


def perturb_bucketize_some(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    float_cols = [c for c in df.columns if "_float" in c]
    rng.shuffle(float_cols)
    for c in float_cols[: max(1, int(0.3 * len(float_cols)))]:
        try:
            binned = pd.qcut(
                pd.to_numeric(df[c], errors="coerce"),
                q=rng.randint(3, 10),
                duplicates="drop",
            )
            df[c] = binned.astype(str)
            df.rename(columns={c: c.replace("_float", "_bucket")}, inplace=True)
        except Exception:
            pass
    return df


PERTURB_MAP = {
    "shuffle_rows": perturb_shuffle_rows,
    "permute_cols": perturb_permute_cols,
    "inject_rare_tokens": perturb_inject_rare_tokens,
    "noise_numeric": perturb_noise_numeric,
    "bucketize_some": perturb_bucketize_some,
}


def apply_hash_perturbations(df: pd.DataFrame, cfg: Dict, tag: str = "") -> pd.DataFrame:
    if not cfg["enable_hash_perturb"]:
        return df
    for i in range(random.randint(*cfg["hash_iterations"])):
        algo = random.choice(cfg["hash_algos"])
        seed = hash_to_seed(json.dumps({"shape": df.shape, "tag": tag, "i": i}), algo)
        rng = random.Random(seed)
        for op in random.sample(cfg["perturb_ops"], random.randint(1, 3)):
            df = PERTURB_MAP[op](df, rng)
    return df


# =========================
# Main generation
# =========================

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def random_size(rr: Tuple[int, int], cr: Tuple[int, int]) -> Tuple[int, int]:
    return random.randint(*rr), random.randint(*cr)


def main():
    cfg = EVAL_CONFIG
    
    # Create output directory
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["log_filename"]), exist_ok=True)
    
    # Use different seed from training to ensure different data
    EVAL_SEED = 0xDEADBEEF2026
    random.seed(EVAL_SEED)
    np.random.seed(EVAL_SEED % (2**32 - 1))
    
    print("=" * 60)
    print("EVALUATION DATASET GENERATION")
    print("=" * 60)
    print(f"Number of datasets: {cfg['num_datasets']}")
    print(f"Rows range: {cfg['rows_range'][0]}-{cfg['rows_range'][1]}")
    print(f"Cols range: {cfg['cols_range'][0]}-{cfg['cols_range'][1]}")
    print(f"Output directory: {cfg['output_dir']}")
    print(f"Log file: {cfg['log_filename']}")
    print("=" * 60)
    
    log: List[Dict] = []
    
    for i in range(1, cfg["num_datasets"] + 1):
        rows, cols = random_size(cfg["rows_range"], cfg["cols_range"])
        
        # Set per-dataset seed
        ds_seed = hash_to_seed(f"eval_{i}_{EVAL_SEED}")
        random.seed(ds_seed)
        np.random.seed(ds_seed % (2**32 - 1))
        
        # Generate dataset
        df = generate_base_dataset(rows, cols, cfg)
        df = apply_hash_perturbations(df, cfg, f"{cfg['prefix']}_{i}")
        
        # Save
        name = f"{cfg['prefix']}_{i:04d}_{rows}x{cols}.csv"
        path = os.path.join(cfg["output_dir"], name)
        df.to_csv(path, index=False)
        
        dataset_hash = file_sha256(path)
        log.append({
            "id": i,
            "file": name,
            "rows": rows,
            "cols": cols,
            "hash": dataset_hash[:16],
            "hash_full": dataset_hash,
            "created": datetime.now(UTC).isoformat(),
        })
        
        if i % 10 == 0:
            print(f"  Progress: {i}/{cfg['num_datasets']} datasets generated...")
    
    # Save log
    log_df = pd.DataFrame(log)
    log_df.to_csv(cfg["log_filename"], index=False)
    
    print(f"\n✅ Generated {len(log)} evaluation datasets")
    print(f"   Directory: {cfg['output_dir']}")
    print(f"   Log file: {cfg['log_filename']}")


if __name__ == "__main__":
    main()
