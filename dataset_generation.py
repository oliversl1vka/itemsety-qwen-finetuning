
"""Synthetic Dataset Factory for Frequent Itemset Mining.

Generates multiple synthetic CSV datasets with mixed column types (int, float,
text, bool) and applies optional perturbations for variability. Output is
organized into an output directory with a nested data subfolder. Designed to
create diverse inputs for a frequent itemset mining pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import string
from datetime import datetime, UTC
from pathlib import Path
from secrets import token_hex
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd



# =========================

# CONFIGURATION

# =========================

CONFIG = {
    # Counts / sizes
    "num_datasets": 90,  # Number of datasets per run
    "rows_range": (5, 100),  # (min_rows, max_rows)
    "cols_range": (10, 100),  # (min_cols, max_cols)
    # Probabilities for column data types
    "col_type_probs": {"int": 0.35, "float": 0.25, "text": 0.30, "bool": 0.10},
    # Numeric ranges
    "int_range": (-10_000, 10_000),
    "float_range": (-1e6, 1e6),
    "float_decimals": 4,
    # Text tokens and synthetic word generation
    "num_synthetic_words": 50_000,
    "curated_tokens": {
        "colors": [
            "red",
            "green",
            "blue",
            "yellow",
            "black",
            "white",
            "orange",
            "purple",
            "pink",
            "brown",
            "gray",
        ],
        "animals": ["ant", "bee", "cat", "dog", "fox", "owl", "yak", "zebra"],
        "products": [
            "phone",
            "laptop",
            "mug",
            "bottle",
            "chair",
            "lamp",
            "watch",
            "backpack",
        ],
        "cities": [
            "prague",
            "vienna",
            "berlin",
            "paris",
            "london",
            "madrid",
            "rome",
        ],
    },
    "frequent_token_bias": 0.15,
    # Hash-based perturbations
    "enable_hash_perturb": True,
    "hash_iterations": (1, 3),
    "hash_algos": ["md5", "sha1", "sha256", "blake2b", "blake2s"],
    "perturb_ops": [
        "shuffle_rows",
        "permute_cols",
        "inject_rare_tokens",
        "noise_numeric",
        "bucketize_some",
    ],
    # Output settings (adjusted to integrate with main project)
    # Now writing directly into existing project 'datasets' directory used by pipeline.
    # Leave data_subfolder empty to avoid nested structure.
    "output_dir": "datasets",
    "data_subfolder": "",  # empty => no nested subfolder
    "prefix": "ds",
    "log_filename": "generation_log.csv",
}



# =========================

# Utility functions

# =========================



def hash_to_seed(s: str, algo: str = "sha256") -> int:
    """Derive deterministic seed from a hash string."""
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

def generate_base_dataset(rows: int, cols: int, cfg: Dict[str, any] = CONFIG) -> pd.DataFrame:  # type: ignore[name-defined]
    col_types = draw_col_types(cols, cfg["col_type_probs"])
    lexicon = build_text_lexicon(cfg["num_synthetic_words"], cfg["curated_tokens"])
    data: Dict[str, List[str]] = {}
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
            span = int(abs(CONFIG["int_range"][1] - CONFIG["int_range"][0]) * noise_level)
            noise = np.random.randint(-span, span + 1, len(df))
            df[c] = df[c].astype(int) + noise
        else:
            span = abs(CONFIG["float_range"][1] - CONFIG["float_range"][0]) * noise_level
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



def apply_hash_perturbations(df: pd.DataFrame, cfg: Dict[str, any] = CONFIG, tag: str = "") -> pd.DataFrame:  # type: ignore[name-defined]
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

# Main generation loop

# =========================

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# Compute where to place datasets. If data_subfolder is empty, use output_dir directly.
if CONFIG["data_subfolder"]:
    data_dir = os.path.join(CONFIG["output_dir"], CONFIG["data_subfolder"])
    os.makedirs(data_dir, exist_ok=True)
else:
    data_dir = CONFIG["output_dir"]



log: List[Dict[str, any]] = []


def random_size(rr: Tuple[int, int], cr: Tuple[int, int]) -> Tuple[int, int]:
    return random.randint(*rr), random.randint(*cr)

def _load_existing_state(cfg: Dict[str, any]):  # type: ignore[name-defined]
    """Return (start_index, existing_log_df or None).

    Tries to read existing log; if present, derives max id. Fallback: scan filenames.
    """
    # NOTE: log file now stored at project root, not inside datasets directory
    log_path = cfg["log_filename"]
    if os.path.isfile(log_path):
        try:
            existing = pd.read_csv(log_path)
            if "id" in existing.columns and len(existing):
                return int(existing["id"].max()) + 1, existing
        except Exception:
            pass
    # fallback: scan directory for pattern prefix_XXXX_ and extract numbers
    max_found = 0
    for fname in os.listdir(cfg["output_dir"]):
        if fname.startswith(f"{cfg['prefix']}_") and fname.endswith('.csv'):
            # pattern: prefix_XXXX_rowsxcols.csv
            parts = fname.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                num = int(parts[1])
                if num > max_found:
                    max_found = num
    return max_found + 1, None


# derive starting global index
_start_index, _existing_log = _load_existing_state(CONFIG)


# =========================
# Master seed + hashing helpers
# =========================

MASTER_SEED_FILE = "master_seed.txt"


def load_or_create_master_seed(path: str = MASTER_SEED_FILE) -> int:
    if os.path.isfile(path):
        try:
            txt = Path(path).read_text(encoding="utf-8").strip()
            return int(txt, 16)
        except Exception:
            pass
    seed_hex = token_hex(16)  # 128 bits
    Path(path).write_text(seed_hex, encoding="utf-8")
    return int(seed_hex, 16)


MASTER_SEED = load_or_create_master_seed()
random.seed(MASTER_SEED + _start_index)
np.random.seed((MASTER_SEED + _start_index) % (2**32 - 1))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def allocate_next_id(cfg: Dict[str, any]) -> int:
    """Re-scan log and filesystem under lock to avoid ID collisions."""
    log_path = os.path.join(cfg["output_dir"], cfg["log_filename"])
    max_id = 0
    if os.path.isfile(log_path):
        try:
            existing = pd.read_csv(log_path, usecols=["id"])
            if len(existing):
                max_id = int(existing["id"].max())
        except Exception:
            pass
    for fname in os.listdir(cfg["output_dir"]):
        if fname.startswith(f"{cfg['prefix']}_") and fname.endswith('.csv'):
            parts = fname.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                num = int(parts[1])
                if num > max_id:
                    max_id = num
    return max_id + 1


CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()


def per_dataset_seed(global_id: int) -> int:
    # Derive deterministic per-dataset seed from master seed + id + config hash fragment
    mix = f"{MASTER_SEED}:{global_id}:{CONFIG_HASH[:16]}".encode()
    return int(hashlib.sha256(mix).hexdigest()[:16], 16)


def generate_and_save_global(global_id: int, cfg: Dict[str, any] = CONFIG) -> str:  # type: ignore[name-defined]
    rows, cols = random_size(cfg["rows_range"], cfg["cols_range"])
    # Set per-dataset seeds (local randomness isolation)
    ds_seed = per_dataset_seed(global_id)
    random.seed(ds_seed)
    np.random.seed(ds_seed % (2**32 - 1))
    df = generate_base_dataset(rows, cols, cfg)
    df = apply_hash_perturbations(df, cfg, f"{cfg['prefix']}_{global_id}")
    name = f"{cfg['prefix']}_{global_id:04d}_{rows}x{cols}.csv"
    path = os.path.join(data_dir, name)
    if os.path.exists(path):
        # Extremely unlikely if allocate_next_id works, fallback add suffix until unique
        suffix = 1
        base_name, ext = os.path.splitext(name)
        while os.path.exists(os.path.join(data_dir, f"{base_name}__{suffix}{ext}")):
            suffix += 1
        name = f"{base_name}__{suffix}{ext}"
        path = os.path.join(data_dir, name)
    df.to_csv(path, index=False)
    dataset_hash = file_sha256(path)
    log.append(
        {
            "id": global_id,
            "file": name,
            "rows": rows,
            "cols": cols,
            "hash": dataset_hash[:16],
            "hash_full": dataset_hash,
            "created": datetime.now(UTC).isoformat(),
            "master_seed_hex": format(MASTER_SEED, 'x'),
            "config_hash": CONFIG_HASH[:16],
            "dataset_seed_hex": format(ds_seed, 'x'),
        }
    )
    return path


# generate using persistent indices (no locking needed for single-process usage)
paths: List[str] = []
next_id = allocate_next_id(CONFIG)
for _ in range(CONFIG["num_datasets"]):
    paths.append(generate_and_save_global(next_id))
    next_id += 1

# append log safely
log_df = pd.DataFrame(log)
log_path_final = CONFIG["log_filename"]  # root-level generation log
if os.path.isfile(log_path_final):
    # append without header
    log_df.to_csv(log_path_final, mode="a", header=False, index=False)
else:
    log_df.to_csv(log_path_final, index=False)

print(
    f"✅ Generated {len(paths)} datasets → stored in: {data_dir}. Last ID: {log[-1]['id']} "
    f"(config_hash={CONFIG_HASH[:8]} master_seed={format(MASTER_SEED, 'x')[:8]}...)"
)