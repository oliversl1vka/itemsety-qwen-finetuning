"""Simplified frequent itemset pipeline.

Generates Apriori frequent itemsets (deterministic) and optionally LLM-extracted
itemsets over either ALL transactions (chunked) or a sample (single chunk).

Removed: classification / legacy metrics / evidence overlap reporting.
Outputs:
 - apriori_output.json (deterministic Apriori itemsets)
 - extractor_output.json (LLM-extracted itemsets or fallback singletons)
 - validation_report.json (invariant checks)
 - db_prepared.json (run summary with dataset metadata & timing)
Optional persistence:
 - runs.db (SQLite table `runs`) unless --disable-db

Environment variables are loaded from openai.env if present.
"""
from __future__ import annotations
import os, json, re, csv, itertools, argparse, sys, sqlite3, time, hashlib, platform
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv

#############################################
# Logging path helper
#############################################
LOG_ROOT = Path('logs')
LOG_KINDS = {
    'apriori': 'apriori',
    'extractor': 'extractor',
    'validation': 'validation',
    'db_prepared': 'db_prepared'
}

def resolve_log_path(kind: str, stem: str, hash_prefix: str, suffix: str = "", model_prefix: str = "") -> Path:
    """Return path under logs/<kind>/ for generation log JSON for a given dataset stem/hash.
    suffix: extra string (already starting with _ if desired) appended before .json for uniqueness (e.g. timestamp).
    model_prefix: model identifier prefix (e.g. 'gpt_4_1_') to prepend to filename.
    Naming pattern: <model_prefix><kind>_generation_log_<stem>_<hash><suffix>.json
    Unrecognized kind falls back to 'misc'.
    """
    safe_kind = LOG_KINDS.get(kind, 'misc')
    subdir = LOG_ROOT / safe_kind
    filename = f"{model_prefix}{kind}_generation_log_{stem}_{hash_prefix}{suffix}.json"
    return subdir / filename

#############################
# Validation Helpers (placed early to avoid NameError)
#############################

def build_row_item_map(transactions: List[List[str]]) -> Dict[str, set]:
    """Map 'Row N' -> set of canonical items present in that transaction."""
    mapping: Dict[str, set] = {}
    for idx, trans in enumerate(transactions):
        label = f'Row {idx+1}'
        canon_items = set(str(x).strip().lower() for x in trans if str(x).strip())
        mapping[label] = canon_items
    return mapping

def canonical_itemset(itemset: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(str(x).strip().lower() for x in itemset if str(x).strip()))

def validate_source(itemsets: List[Dict[str, Any]], row_item_map: Dict[str, set], total_rows: int,
                    min_support: int, source_name: str) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    validated = 0
    for rec in itemsets:
        its = rec.get('itemset')
        context = {'source': source_name, 'raw_itemset': its}
        rows = rec.get('rows') or rec.get('evidence_rows') or []
        count = rec.get('count')
        # Structural checks
        if not isinstance(its, (list, tuple)) or not its:
            errors.append({'type': 'empty_itemset', **context})
            continue
        if not isinstance(rows, list) or not rows:
            errors.append({'type': 'missing_rows', **context})
            continue
        canon = canonical_itemset(list(its))
        # Normalize row labels
        norm_rows: List[str] = []
        for r in rows:
            if isinstance(r, str) and r.lower().startswith('row '):
                try:
                    idx = int(r.split()[1])
                    norm_rows.append(f'Row {idx}')
                except Exception:
                    errors.append({'type': 'malformed_row_label', 'row': r, **context})
            else:
                try:
                    idx = int(str(r))
                    norm_rows.append(f'Row {idx}')
                except Exception:
                    errors.append({'type': 'malformed_row_label', 'row': r, **context})
        unique_rows = sorted(set(norm_rows))
        # Count checks
        if count is None:
            errors.append({'type': 'missing_count', **context})
        elif count != len(unique_rows):
            errors.append({'type': 'count_mismatch', 'reported_count': count, 'unique_row_count': len(unique_rows), **context})
        # Min support check
        if isinstance(count, (int, float)) and count < min_support:
            errors.append({'type': 'below_min_support', 'count': count, 'min_support': min_support, **context})
        # Row range & content checks
        for rl in unique_rows:
            try:
                idx = int(rl.split()[1])
            except Exception:
                errors.append({'type': 'unparseable_row_index', 'row': rl, **context})
                continue
            if idx < 1 or idx > total_rows:
                errors.append({'type': 'row_out_of_range', 'row': rl, 'total_rows': total_rows, **context})
                continue
            present_items = row_item_map.get(rl, set())
            missing = [it for it in canon if it not in present_items]
            if missing:
                errors.append({'type': 'item_missing_in_row', 'row': rl, 'missing_items': missing, **context})
        # Apriori support verification (only for apriori source if field present)
        if source_name == 'apriori' and 'support' in rec and isinstance(count, int):
            expected_support = len(unique_rows) / total_rows if total_rows else 0.0
            reported_support = rec.get('support')
            if not isinstance(reported_support, (int, float)):
                errors.append({'type': 'invalid_support_type', 'reported_support': reported_support, **context})
            else:
                if abs(reported_support - expected_support) > 1e-6:
                    errors.append({'type': 'support_mismatch', 'reported_support': reported_support,
                                   'expected_support': expected_support, **context})
        # Duplicate rows detection
        if len(rows) != len(unique_rows):
            errors.append({'type': 'duplicate_rows_listed', 'listed_rows': rows, 'unique_rows': unique_rows, **context})
        # Validation success only if NO errors for this itemset
        itemset_errors = [e for e in errors if e.get('raw_itemset') == its]
        if not itemset_errors:
            validated += 1
    return {
        'source': source_name,
        'total_itemsets': len(itemsets),
        'validated_itemsets': validated,
        'errors': errors
    }

def validate_all(apriori_sets: List[Dict[str, Any]], llm_sets: List[Dict[str, Any]],
                 transactions: List[List[str]], min_support: int) -> Dict[str, Any]:
    row_item_map = build_row_item_map(transactions)
    apriori_report = validate_source(apriori_sets, row_item_map, len(transactions), min_support, source_name='apriori')
    llm_report = validate_source(llm_sets, row_item_map, len(transactions), min_support, source_name='llm')
    summary = {
        'apriori_valid_ratio': safe_div(apriori_report['validated_itemsets'], apriori_report['total_itemsets']),
        'llm_valid_ratio': safe_div(llm_report['validated_itemsets'], llm_report['total_itemsets']),
        'invariants': [
            'count == number of unique evidence rows',
            'each evidence row exists and contains all items',
            'support matches count/total_rows (Apriori)',
            'no itemset below min_support retained',
            'no malformed row labels',
        ]
    }
    return {'summary': summary, 'apriori': apriori_report, 'llm': llm_report}

def safe_div(a: int, b: int) -> float:
    return round(a / b, 6) if b else 0.0

def compute_dataset_metadata(path: str, transactions: List[List[str]]) -> Dict[str, Any]:
    name = os.path.basename(path)
    size_bytes = os.path.getsize(path) if os.path.exists(path) else 0
    sha = hashlib.sha256()
    try:
        with open(path, 'rb') as fh:
            for chunk in iter(lambda: fh.read(8192), b''):
                sha.update(chunk)
        file_hash = sha.hexdigest()
    except Exception:
        file_hash = 'unavailable'
    dataset_id = f"{name}:{file_hash[:12]}" if file_hash != 'unavailable' else f"{name}:unknown"
    return {
        'dataset_name': name,
        'dataset_hash': file_hash,
        'dataset_id': dataset_id,
        'dataset_size_rows': len(transactions),
        'dataset_size_bytes': size_bytes
    }

def build_run_summary(apriori_sets: List[Dict[str, Any]], llm_sets: List[Dict[str, Any]],
                      validation: Dict[str, Any], args: argparse.Namespace, validation_passed: bool,
                      dataset_meta: Dict[str, Any], run_duration_ms: int,
                      apriori_path: str, llm_path: str, validation_path: str, summary_path: str,
                      error_message: Optional[str] = None) -> Dict[str, Any]:
    summary = {
        'timestamp': datetime.now(UTC).isoformat(),
        'python_version': platform.python_version(),
        'data_path': args.data,
        'dataset_id': dataset_meta['dataset_id'],
        'dataset_name': dataset_meta['dataset_name'],
        'dataset_hash': dataset_meta['dataset_hash'],
        'dataset_size_rows': dataset_meta['dataset_size_rows'],
        'dataset_size_bytes': dataset_meta['dataset_size_bytes'],
        'min_support': args.min_support,
        'max_size': args.max_size,
        'llm_full': bool(args.llm_full),
        'llm_chunk_size': args.llm_chunk_size,
        'llm_model': getattr(args, 'llm_model', 'unknown'),
        'apriori_itemset_count': len(apriori_sets),
        'llm_itemset_count': len(llm_sets),
        'validation_passed': validation_passed,
        'apriori_valid_ratio': validation['summary']['apriori_valid_ratio'],
        'llm_valid_ratio': validation['summary']['llm_valid_ratio'],
        'apriori_errors': len(validation['apriori']['errors']),
        'llm_errors': len(validation['llm']['errors']),
        'invariants': validation['summary']['invariants'],
        'run_duration_ms': run_duration_ms,
        'apriori_output_path': apriori_path,
        'llm_output_path': llm_path,
        'validation_report_path': validation_path,
        'summary_path': summary_path,
        'error_message': error_message
    }
    return summary

def cleanup_generic_outputs(workdir: Path) -> List[str]:
    """Delete generic (non-hash) output artifact files to reduce clutter.
    Returns list of removed file paths.
    Safe: ignores missing files.
    """
    generic = [
        'apriori_output.json',
        'extractor_output.json',
        'validation_report.json',
        'db_prepared.json'
    ]
    removed: List[str] = []
    for name in generic:
        fp = workdir / name
        try:
            if fp.exists():
                fp.unlink()
                removed.append(str(fp))
        except Exception:
            # Non-fatal
            pass
    return removed

def persist_run_to_sqlite(run_summary: Dict[str, Any], db_path: str) -> int:
    schema_sql = (
        "CREATE TABLE IF NOT EXISTS runs ("\
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"\
        "timestamp TEXT,"\
        "python_version TEXT,"\
        "data_path TEXT,"\
        "dataset_id TEXT,"\
        "dataset_name TEXT,"\
        "dataset_hash TEXT,"\
        "dataset_size_rows INTEGER,"\
        "dataset_size_bytes INTEGER,"\
        "min_support INTEGER,"\
        "max_size INTEGER,"\
        "llm_full INTEGER,"\
        "llm_chunk_size INTEGER,"\
        "apriori_itemset_count INTEGER,"\
        "llm_itemset_count INTEGER,"\
        "validation_passed INTEGER,"\
        "apriori_valid_ratio REAL,"\
        "llm_valid_ratio REAL,"\
        "apriori_errors INTEGER,"\
        "llm_errors INTEGER,"\
        "run_duration_ms INTEGER,"\
        "invariants TEXT,"\
        "apriori_output_path TEXT,"\
        "llm_output_path TEXT,"\
        "validation_report_path TEXT"\
        ")"
    )
    new_cols = [
        ('python_version','TEXT'), ('dataset_id','TEXT'), ('dataset_name','TEXT'), ('dataset_hash','TEXT'),
        ('dataset_size_rows','INTEGER'), ('dataset_size_bytes','INTEGER'), ('run_duration_ms','INTEGER'),
        ('error_message','TEXT'), ('llm_model','TEXT')
    ]
    insert_sql = (
        "INSERT INTO runs (timestamp, python_version, data_path, dataset_id, dataset_name, dataset_hash, "
        "dataset_size_rows, dataset_size_bytes, min_support, max_size, llm_full, llm_chunk_size, "
        "apriori_itemset_count, llm_itemset_count, validation_passed, apriori_valid_ratio, llm_valid_ratio, "
        "apriori_errors, llm_errors, run_duration_ms, invariants, apriori_output_path, llm_output_path, validation_report_path, error_message, llm_model) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(schema_sql)
        existing = {r[1] for r in cur.execute("PRAGMA table_info(runs)").fetchall()}
        for col, ctype in new_cols:
            if col not in existing:
                cur.execute(f"ALTER TABLE runs ADD COLUMN {col} {ctype}")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_validation ON runs(validation_passed)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_dataset ON runs(dataset_id)")
        cur.execute(insert_sql, (
            run_summary['timestamp'],
            run_summary['python_version'],
            run_summary['data_path'],
            run_summary['dataset_id'],
            run_summary['dataset_name'],
            run_summary['dataset_hash'],
            run_summary['dataset_size_rows'],
            run_summary['dataset_size_bytes'],
            run_summary['min_support'],
            run_summary['max_size'],
            1 if run_summary['llm_full'] else 0,
            run_summary['llm_chunk_size'],
            run_summary['apriori_itemset_count'],
            run_summary['llm_itemset_count'],
            1 if run_summary['validation_passed'] else 0,
            run_summary['apriori_valid_ratio'],
            run_summary['llm_valid_ratio'],
            run_summary['apriori_errors'],
            run_summary['llm_errors'],
            run_summary['run_duration_ms'],
            json.dumps(run_summary['invariants'], ensure_ascii=False),
            run_summary['apriori_output_path'],
            run_summary['llm_output_path'],
            run_summary['validation_report_path'],
            run_summary.get('error_message'),
            run_summary.get('llm_model', 'unknown')
        ))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()

def load_transactions_csv(path: str) -> Tuple[pd.DataFrame, List[List[str]], Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    sample_lines: List[str] = []
    with p.open('r', encoding='utf-8', errors='ignore') as fh:
        for _ in range(40):
            ln = fh.readline()
            if not ln:
                break
            sample_lines.append(ln)
    cleaned_lines = [ln for ln in sample_lines if ln.strip()]
    try:
        dialect = csv.Sniffer().sniff(''.join(cleaned_lines) or '', delimiters=",;\t|")
        sep_guess = dialect.delimiter
    except Exception:
        sep_guess = ','
    try:
        df = pd.read_csv(path, sep=sep_guess, low_memory=False)
        strategy = 'standard'
    except Exception:
        df = pd.read_csv(path, sep=sep_guess, engine='python', on_bad_lines='skip')
        strategy = 'python-skip'
    lower_cols = [c.lower() for c in df.columns]
    has_tid = any(c in lower_cols for c in ['transaction_id','tid','trans_id','t_id'])
    has_item = any(c in lower_cols for c in ['item','product','sku'])
    if has_tid and has_item:
        fmt = 'long'
    elif df.shape[1] == 1:
        fmt = 'single-column'
    else:
        fmt = 'wide'
    transactions: List[List[str]] = []
    if fmt == 'long':
        cols_map = {c.lower(): c for c in df.columns}
        tid_col = next((cols_map[c] for c in ['transaction_id','tid','trans_id','t_id'] if c in cols_map), None)
        item_col = next((cols_map[c] for c in ['item','product','sku'] if c in cols_map), None)
        if not tid_col or not item_col:
            raise ValueError("Detected 'long' format but needed columns missing")
        grouped = df.groupby(tid_col)[item_col].apply(lambda s: [str(v).strip() for v in s if str(v).strip()])
        transactions = grouped.tolist()
    elif fmt == 'wide':
        for _, row in df.iterrows():
            items = []
            for v in row.tolist():
                if pd.isna(v):
                    continue
                sval = str(v).strip()
                if not sval or sval.lower() in {'nan','none'}:
                    continue
                items.append(sval)
            if items:
                transactions.append(items)
    else:
        for raw in df.iloc[:, 0].astype(str):
            parts = re.split(r'[;,|\t]', raw)
            if len(parts) == 1 and ' ' in raw and ',' not in raw and ';' not in raw:
                parts = raw.split()
            cleaned = [p.strip() for p in parts if p.strip()]
            if cleaned:
                transactions.append(cleaned)
    return df, transactions, {'strategy': strategy, 'format': fmt, 'sep': sep_guess}

def apriori_frequent_itemsets(transactions: List[List[str]], min_support: int = 3, max_size: int = 3) -> List[Dict[str, Any]]:
    if not transactions:
        return []
    row_labels = [f"Row {i+1}" for i in range(len(transactions))]
    counts: Dict[Tuple[str,...], Dict[str, Any]] = {}
    for idx, trans in enumerate(transactions):
        seen = set()
        for item in trans:
            item = str(item).strip().lower()
            if not item or item in seen:
                continue
            seen.add(item)
            k = (item,)
            if k not in counts:
                counts[k] = {'count': 0, 'rows': []}
            counts[k]['count'] += 1
            counts[k]['rows'].append(row_labels[idx])
    def prune(d: Dict[Tuple[str,...], Dict[str, Any]]) -> Dict[Tuple[str,...], Dict[str, Any]]:
        return {k: v for k, v in d.items() if v['count'] >= min_support}
    freq_levels: List[Dict[Tuple[str,...], Dict[str, Any]]] = []
    L1 = prune(counts)
    if not L1:
        return []
    freq_levels.append(L1)
    current = L1
    k = 2
    while k <= max_size and current:
        prev_keys = sorted(current.keys())
        candidates: set[Tuple[str,...]] = set()
        for i in range(len(prev_keys)):
            for j in range(i+1, len(prev_keys)):
                a = prev_keys[i]; b = prev_keys[j]
                if a[:k-2] == b[:k-2]:
                    merged = tuple(sorted(set(a) | set(b)))
                    if len(merged) == k:
                        if all(tuple(sorted(sub)) in current for sub in itertools.combinations(merged, k-1)):
                            candidates.add(merged)
        if not candidates:
            break
        cand_counts = {c: {'count': 0, 'rows': []} for c in candidates}
        for idx, trans in enumerate(transactions):
            tset = set(map(lambda x: str(x).strip().lower(), trans))
            for cand in candidates:
                if set(cand).issubset(tset):
                    cand_counts[cand]['count'] += 1
                    cand_counts[cand]['rows'].append(row_labels[idx])
        current = prune(cand_counts)
        if current:
            freq_levels.append(current)
        k += 1
    out = []
    total = len(transactions)
    for level in freq_levels:
        for itemset, info in level.items():
            rows = info['rows']
            count = info['count']
            out.append({
                'itemset': list(itemset),
                'count': count,
                'rows': rows,
                'unique_rows': sorted(set(rows)),
                'unique_row_count': len(set(rows)),
                'support': count / total if total else None,
                'size': len(itemset)
            })
    return out


def llm_extract_full(transactions: List[List[str]], min_support: int, system_prompt: str, chunk_size: int,
                     api_key: str, model: str) -> List[Dict[str, Any]]:
    """Extract frequent itemsets using OpenAI Chat Completions API directly."""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY - cannot proceed with LLM extraction")
    
    # Detect if this is a reasoning model (o-series, o1, o3, o4)
    # Reasoning models don't support temperature parameter
    is_reasoning_model = any(keyword in model.lower() for keyword in ['o1-', 'o3-', 'o4-', 'o-series'])
    
    # Build LLM kwargs based on model type
    llm_kwargs = {
        'api_key': api_key,
        'model': model,
    }
    
    # Only add temperature for non-reasoning models
    if not is_reasoning_model:
        llm_kwargs['temperature'] = 0.0
    
    llm = ChatOpenAI(**llm_kwargs)
    template = (
        "{system}\nMin support count: {min_support}\nTransactions chunk rows {start}-{end}:\n{chunk_json}\n"
        "Return ONLY JSON array: [{{'itemset':[...],'count':n,'evidence_rows':[...]}}] with count >= {min_support}."
    )
    # Prompt & output parser already imported above
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    aggregated: Dict[Tuple[str,...], Dict[str, Any]] = {}
    total_rows = len(transactions)
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunk = transactions[start:end]
        raw = chain.invoke({
            'system': system_prompt,
            'min_support': min_support,
            'start': start + 1,
            'end': end,
            'chunk_json': json.dumps(chunk, ensure_ascii=False)
        })
        try:
            parsed = json.loads(raw)
        except Exception:
            m = re.search(r'\[[\s\S]*\]', raw)
            parsed = json.loads(m.group(0)) if m else []
        for rec in parsed:
            if not isinstance(rec, dict) or 'itemset' not in rec:
                continue
            its_raw = rec['itemset']
            if not isinstance(its_raw, (list, tuple)) or not its_raw:
                continue
            canon = tuple(sorted(str(x).strip().lower() for x in its_raw))
            ev = rec.get('evidence_rows') or rec.get('rows') or []
            if not isinstance(ev, list):
                ev = []
            norm_rows = []
            for r in ev:
                if isinstance(r, str) and r.lower().startswith('row '):
                    try:
                        idx = int(r.split()[1])
                        norm_rows.append(f'Row {idx}')
                    except Exception:
                        norm_rows.append(str(r))
                else:
                    try:
                        idx = int(str(r))
                        norm_rows.append(f'Row {idx}')
                    except Exception:
                        norm_rows.append(str(r))
            aggregated.setdefault(canon, {'rows': set()})['rows'].update(norm_rows)
    out: List[Dict[str, Any]] = []
    for its, info in aggregated.items():
        rows_set = sorted(info['rows'])
        if len(rows_set) >= min_support:
            out.append({'itemset': list(its), 'count': len(rows_set), 'rows': rows_set})
    return out

    # (Removed legacy duplicate build_run_summary & persist_run_to_sqlite definitions; using enhanced versions above.)

def main():
    parser = argparse.ArgumentParser(description="Frequent itemset pipeline (Apriori + LLM, simplified)")
    parser.add_argument('--data', default='dataset.csv', help='Single dataset CSV path (used if --data-dir not set)')
    parser.add_argument('--data-dir', help='Directory containing multiple CSV datasets to process sequentially')
    parser.add_argument('--min-support', type=int, default=int(os.getenv('MIN_SUPPORT_COUNT', '3')))
    parser.add_argument('--max-size', type=int, default=int(os.getenv('APRIORI_MAX_SIZE', '3')))
    parser.add_argument('--output-apriori', default='apriori_output.json')
    parser.add_argument('--output-llm', default='extractor_output.json')
    parser.add_argument('--llm-full', action='store_true', help='Use ALL rows for LLM extraction (chunked)')
    parser.add_argument('--llm-chunk-size', type=int, default=50)
    parser.add_argument('--llm-model', default=os.getenv('LLM_MODEL','gpt-4o'), help='OpenAI model name (default env LLM_MODEL or gpt-4o)')
    parser.add_argument('--system-prompt', default='extractor_system_prompt.md')
    parser.add_argument('--sqlite-db', default='runs.db', help='SQLite database file path')
    parser.add_argument('--disable-db', action='store_true', help='Disable SQLite persistence')
    parser.add_argument('--cleanup-old', action='store_true', help='Remove generic output files (non-hash) after each run')
    parser.add_argument('--artifact-mode', choices=['overwrite','timestamp'], default='overwrite',
                        help='overwrite: reuse hash-based filenames (default). timestamp: append run timestamp for unique artifacts every run.')
    args = parser.parse_args()

    load_dotenv('openai.env')  # Load OPENAI_API_KEY from openai.env if present
    start_time = time.perf_counter()
    # Support batch mode
    data_files: List[str]
    if args.data_dir:
        if not os.path.isdir(args.data_dir):
            print(f"Specified --data-dir '{args.data_dir}' is not a directory", file=sys.stderr)
            sys.exit(2)
        data_files = [str(Path(args.data_dir)/f) for f in sorted(os.listdir(args.data_dir)) if f.lower().endswith('.csv')]
        if not data_files:
            print(f"No CSV files found in {args.data_dir}")
            sys.exit(0)
    else:
        data_files = [args.data]

    # OpenAI API credentials
    api_key = os.getenv('OPENAI_API_KEY', '')
    model = args.llm_model  # e.g. gpt-4o, gpt-4-turbo, gpt-3.5-turbo
    # Enforce presence of OpenAI API key
    if not api_key:
        print("Missing OPENAI_API_KEY environment variable. Aborting.", file=sys.stderr)
        sys.exit(3)
    try:
        system_prompt_text = Path(args.system_prompt).read_text(encoding='utf-8').strip() or 'You are a frequent itemset extractor. Return JSON array.'
    except Exception:
        system_prompt_text = 'You are a frequent itemset extractor. Return JSON array.'

    total_files = len(data_files)
    print(f"Processing {total_files} dataset(s)...")
    successes = 0
    failures = 0
    # Ensure log directories exist
    for k in LOG_KINDS.values():
        (LOG_ROOT / k).mkdir(parents=True, exist_ok=True)

    for idx, data_path in enumerate(data_files, start=1):
        print(f"\n=== [{idx}/{total_files}] Dataset: {data_path} ===")
        single_start = time.perf_counter()
        args.data = data_path  # mutate for summary function reuse
        error_message: Optional[str] = None
        apriori_sets: List[Dict[str, Any]] = []
        llm_sets: List[Dict[str, Any]] = []
        validation: Dict[str, Any] = {'summary': {'apriori_valid_ratio':0,'llm_valid_ratio':0,'invariants':[]}, 'apriori': {'errors': [], 'validated_itemsets':0,'total_itemsets':0}, 'llm': {'errors': [], 'validated_itemsets':0,'total_itemsets':0}}
        validation_passed = False
        try:
            df, transactions, meta = load_transactions_csv(data_path)
            print(f"Loaded CSV format={meta['format']} sep='{meta['sep']}' transactions={len(transactions)}")
            dataset_meta = compute_dataset_metadata(data_path, transactions)
            hash_prefix = dataset_meta['dataset_hash'][:12] if dataset_meta['dataset_hash'] != 'unavailable' else 'unknown'
            stem = Path(data_path).stem
            # Per-run timestamp (UTC) for uniqueness when artifact-mode=timestamp
            run_ts = datetime.now(UTC).strftime('%Y%m%d%H%M%S')
            suffix = f"_{run_ts}" if args.artifact_mode == 'timestamp' else ""
            # Model prefix for filenames
            model_prefix = f"{args.llm_model}_"
            # Ensure output directories exist (under artifacts/)
            artifacts_root = Path('artifacts')
            apriori_dir = artifacts_root / 'apriori_outputs'; apriori_dir.mkdir(parents=True, exist_ok=True)
            extractor_dir = artifacts_root / 'extractor_outputs'; extractor_dir.mkdir(parents=True, exist_ok=True)
            summary_dir = artifacts_root / 'db_prepared'; summary_dir.mkdir(parents=True, exist_ok=True)
            validation_dir = artifacts_root / 'validation_reports'; validation_dir.mkdir(parents=True, exist_ok=True)
            apriori_file = f"{model_prefix}apriori_output_{stem}_{hash_prefix}{suffix}.json"
            llm_file = f"{model_prefix}extractor_output_{stem}_{hash_prefix}{suffix}.json"
            validation_file = f"{model_prefix}validation_report_{stem}_{hash_prefix}{suffix}.json"
            summary_file = f"{model_prefix}db_prepared_{stem}_{hash_prefix}{suffix}.json"
            # Log file paths (generation logs) under logs/ (mirror artifact suffix for uniqueness)
            apriori_log_path = resolve_log_path('apriori', stem, hash_prefix, suffix, model_prefix)
            extractor_log_path = resolve_log_path('extractor', stem, hash_prefix, suffix, model_prefix)
            validation_log_path = resolve_log_path('validation', stem, hash_prefix, suffix, model_prefix)
            summary_log_path = resolve_log_path('db_prepared', stem, hash_prefix, suffix, model_prefix)
            apriori_path_full = str(apriori_dir / apriori_file)
            llm_path_full = str(extractor_dir / llm_file)
            validation_path_full = str(validation_dir / validation_file)
            summary_path_full = str(summary_dir / summary_file)
            apriori_sets = apriori_frequent_itemsets(transactions, args.min_support, args.max_size)
            print(f"Apriori found {len(apriori_sets)} itemsets (count >= {args.min_support}).")
            with open(apriori_path_full, 'w', encoding='utf-8') as f:
                json.dump(apriori_sets, f, ensure_ascii=False, indent=2)
            # Write apriori generation log (minimal - only unique timing info)
            try:
                apriori_log = {
                    'stage_duration_ms': int((time.perf_counter() - single_start) * 1000),
                    'itemsets_found': len(apriori_sets),
                    'timestamp': datetime.now(UTC).isoformat(),
                }
                apriori_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(apriori_log_path, 'w', encoding='utf-8') as lf:
                    json.dump(apriori_log, lf, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"(Warn) Failed to write apriori generation log: {e}")
            target_transactions = transactions if args.llm_full else transactions[:min(args.llm_chunk_size, len(transactions))]
            scope = 'ALL' if args.llm_full else f'SAMPLE({len(target_transactions)})'
            print(f"Running LLM extraction over {scope} transactions (chunk_size={args.llm_chunk_size}) ...")
            llm_extraction_start = time.perf_counter()
            llm_sets = llm_extract_full(target_transactions, args.min_support, system_prompt_text,
                                        args.llm_chunk_size, api_key, model)
            llm_extraction_duration = int((time.perf_counter() - llm_extraction_start) * 1000)
            print(f"LLM produced {len(llm_sets)} itemsets.")
            with open(llm_path_full, 'w', encoding='utf-8') as f:
                json.dump(llm_sets, f, ensure_ascii=False, indent=2)
            # Write extractor generation log (minimal - same format as apriori)
            try:
                extractor_log = {
                    'stage_duration_ms': llm_extraction_duration,
                    'itemsets_found': len(llm_sets),
                    'timestamp': datetime.now(UTC).isoformat(),
                }
                extractor_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(extractor_log_path, 'w', encoding='utf-8') as lf:
                    json.dump(extractor_log, lf, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"(Warn) Failed to write extractor generation log: {e}")
            validation = validate_all(apriori_sets, llm_sets, transactions, args.min_support)
            with open(validation_path_full, 'w', encoding='utf-8') as vf:
                json.dump(validation, vf, ensure_ascii=False, indent=2)
            # Validation generation log (minimal - only errors if present)
            try:
                val_log = {
                    'apriori_errors': len(validation['apriori']['errors']),
                    'llm_errors': len(validation['llm']['errors']),
                    'timestamp': datetime.now(UTC).isoformat(),
                }
                # Add first 3 error details if any exist (for quick debugging)
                if validation['apriori']['errors']:
                    val_log['apriori_error_sample'] = validation['apriori']['errors'][:3]
                if validation['llm']['errors']:
                    val_log['llm_error_sample'] = validation['llm']['errors'][:3]
                validation_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(validation_log_path, 'w', encoding='utf-8') as lf:
                    json.dump(val_log, lf, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"(Warn) Failed to write validation generation log: {e}")
            apr_err = len(validation['apriori']['errors'])
            llm_err = len(validation['llm']['errors'])
            validation_passed = apr_err == 0 and llm_err == 0
            if validation_passed:
                print("Validation PASSED.")
            else:
                print(f"Validation FAILED: Apriori errors={apr_err}, LLM errors={llm_err}.")
        except Exception as e:
            error_message = f"{type(e).__name__}: {e}"
            print(f"Error processing {data_path}: {error_message}", file=sys.stderr)
        # Derive filenames even on error if not already set
        if 'dataset_meta' not in locals():
            dataset_meta = compute_dataset_metadata(data_path, [])
            hash_prefix = dataset_meta['dataset_hash'][:12] if dataset_meta['dataset_hash'] != 'unavailable' else 'unknown'
            stem = Path(data_path).stem
            run_ts = datetime.now(UTC).strftime('%Y%m%d%H%M%S')
            suffix = f"_{run_ts}" if args.artifact_mode == 'timestamp' else ""
            # Model prefix for filenames
            model_prefix = f"{args.llm_model}_"
            artifacts_root = Path('artifacts')
            apriori_dir = artifacts_root / 'apriori_outputs'; apriori_dir.mkdir(parents=True, exist_ok=True)
            extractor_dir = artifacts_root / 'extractor_outputs'; extractor_dir.mkdir(parents=True, exist_ok=True)
            summary_dir = artifacts_root / 'db_prepared'; summary_dir.mkdir(parents=True, exist_ok=True)
            validation_dir = artifacts_root / 'validation_reports'; validation_dir.mkdir(parents=True, exist_ok=True)
            apriori_file = f"{model_prefix}apriori_output_{stem}_{hash_prefix}{suffix}.json"
            llm_file = f"{model_prefix}extractor_output_{stem}_{hash_prefix}{suffix}.json"
            validation_file = f"{model_prefix}validation_report_{stem}_{hash_prefix}{suffix}.json"
            summary_file = f"{model_prefix}db_prepared_{stem}_{hash_prefix}{suffix}.json"
            apriori_path_full = str(apriori_dir / apriori_file)
            llm_path_full = str(extractor_dir / llm_file)
            validation_path_full = str(validation_dir / validation_file)
            summary_path_full = str(summary_dir / summary_file)
        # Write validation file even if validation did not run (minimal stub)
        if 'validation' in locals():
            try:
                with open(validation_path_full, 'w', encoding='utf-8') as vf:
                    json.dump(validation, vf, ensure_ascii=False, indent=2)
            except Exception:
                pass
        run_duration_ms = int((time.perf_counter() - single_start) * 1000)
        run_summary = build_run_summary(
            apriori_sets,
            llm_sets,
            validation,
            args,
            validation_passed,
            dataset_meta,
            run_duration_ms,
            apriori_path_full,
            llm_path_full,
            validation_path_full,
            summary_path_full,
            error_message
        )
        with open(summary_path_full, 'w', encoding='utf-8') as rs:
            json.dump(run_summary, rs, ensure_ascii=False, indent=2)
        # Summary generation log (minimal - only failures/errors)
        try:
            summary_log = {
                'validation_passed': validation_passed,
                'run_duration_ms': run_duration_ms,
                'timestamp': datetime.now(UTC).isoformat(),
            }
            # Only add error message if present
            if error_message:
                summary_log['error'] = error_message
            summary_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_log_path, 'w', encoding='utf-8') as lf:
                json.dump(summary_log, lf, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"(Warn) Failed to write summary generation log: {e}")
        print(f"Run summary written to {summary_path_full}")
        if args.cleanup_old:
            removed = cleanup_generic_outputs(Path('.'))
            if removed:
                print(f"Removed generic artifacts: {', '.join(removed)}")
        if not args.disable_db:
            try:
                run_id = persist_run_to_sqlite(run_summary, args.sqlite_db)
                print(f"Persisted run id={run_id} -> {args.sqlite_db}")
            except Exception as db_e:
                print(f"SQLite persistence error: {db_e}", file=sys.stderr)
        else:
            print("(DB disabled)")
        if validation_passed and not error_message:
            successes += 1
        else:
            failures += 1
    print(f"\nBatch complete: {successes} success, {failures} failure(s).")
    # Exit code 0 even if some failed (since user wants full traversal); non-zero only if all failed
    if successes == 0 and failures > 0:
        sys.exit(1)
    print("Simplified pipeline complete.")

if __name__ == '__main__':
    main()
