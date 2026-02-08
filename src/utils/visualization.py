from __future__ import annotations
"""Focused comparative visualization for runs.db.

Outputs ONLY comparative Apriori vs LLM itemset analytics:
    1. grouped_counts_<model>_<tag>.png
    2. diff_bins_<model>_<tag>.png
    3. percent_match_bins_<model>_<tag>.png
    4. percent_match_trend_<model>_<tag>.png
    5. summary_table_<model>_<tag>.csv

Args:
    --db            Path to runs.db
    --limit         Only first N rows chronologically
    --outdir        Output directory (default visuals)
    --top           Top N datasets by area for grouped counts
    --bins          Number of quantile bins (default 5)
    --bin-edges     Explicit numeric edges for area bins
    --model         Filter runs by llm_model
    --only-passed   Restrict to validation_passed=1 runs
    --since         Filter timestamp >= since (ISO)
    --until         Filter timestamp <= until (ISO)
    --date          Filter specific YYYY-MM-DD (ignores since/until)
    --tag           Tag suffix for output filenames; default UTC timestamp
    --show          Show interactive plots
"""
import argparse, os, sqlite3, math, re
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RELEVANT_COLS = [
    "id","timestamp","dataset_name","dataset_size_rows","dataset_size_bytes",
    "min_support","max_size","apriori_itemset_count","llm_itemset_count",
    "apriori_valid_ratio","llm_valid_ratio","apriori_errors","llm_errors",
    "run_duration_ms","error_message","llm_model","validation_passed"
]


def load_runs(db_path: str, limit: int | None, model: str | None, only_passed: bool,
              since: str | None, until: str | None, on_date: str | None) -> pd.DataFrame:
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        where_clauses = []
        params: list = []
        if model:
            where_clauses.append("llm_model = ?")
            params.append(model)
        if only_passed:
            where_clauses.append("validation_passed = 1")
        if on_date:
            day_start = f"{on_date}T00:00:00"
            where_clauses.append("timestamp >= ? AND timestamp < datetime(?, '+1 day')")
            params.extend([day_start, day_start])
        else:
            if since:
                where_clauses.append("timestamp >= ?")
                params.append(since)
            if until:
                where_clauses.append("timestamp <= ?")
                params.append(until)
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        sql = f"SELECT * FROM runs {where_sql} ORDER BY timestamp ASC"
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    cols = [c for c in RELEVANT_COLS if c in df.columns]
    df = df[cols]
    if limit:
        df = df.head(limit)
    if {"apriori_itemset_count","llm_itemset_count","dataset_size_rows"}.issubset(df.columns):
        denom = df["apriori_itemset_count"].replace(0, pd.NA)
        ratio = df["llm_itemset_count"] / denom
        ratio = pd.to_numeric(ratio, errors='coerce')
        ratio.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        df["llm_to_apriori_itemset_ratio"] = ratio
    return df


def ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(r"(\d+)x(\d+)(?:\.csv)?$")
    rows_list, cols_list = [], []
    for name in df.get("dataset_name", pd.Series(dtype=str)).astype(str):
        m = pattern.search(name)
        if m:
            rows_list.append(int(m.group(1)))
            cols_list.append(int(m.group(2)))
        else:
            rows_list.append(pd.NA)
            cols_list.append(pd.NA)
    df["parsed_rows"] = rows_list
    df["parsed_cols"] = cols_list
    # Pandas 3.0+ handles inf values differently
    area = pd.to_numeric(df["parsed_rows"], errors='coerce') * pd.to_numeric(df["parsed_cols"], errors='coerce')
    df["dataset_area"] = area.replace([float('inf'), float('-inf')], pd.NA)
    return df


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if {"apriori_itemset_count","llm_itemset_count"}.issubset(df.columns):
        apr = df["apriori_itemset_count"].replace(0, pd.NA)
        df["itemset_diff"] = df["llm_itemset_count"] - df["apriori_itemset_count"]
        df["percent_match"] = pd.to_numeric(df["llm_itemset_count"] / apr, errors='coerce')
    return df


def assign_bins(df: pd.DataFrame, bins: int | None, edges: list[float] | None) -> pd.DataFrame:
    if "dataset_area" not in df.columns:
        df["size_bin"] = pd.NA
        return df
    try:
        area = pd.to_numeric(df["dataset_area"], errors='coerce').dropna()
    except Exception:
        area = df["dataset_area"].dropna()
    if area.empty:
        df["size_bin"] = pd.NA
        return df
    if edges:
        try:
            edges = sorted(set(float(e) for e in edges))
            if len(edges) < 2:
                raise ValueError("Need at least two edges")
            df["size_bin"] = pd.cut(pd.to_numeric(df["dataset_area"], errors='coerce'), bins=edges, include_lowest=True)
        except Exception:
            edges = None
    if not edges:
        q = bins or 5
        try:
            df["size_bin"] = pd.qcut(pd.to_numeric(df["dataset_area"], errors='coerce'), q=q, duplicates='drop')
        except ValueError:
            df["size_bin"] = pd.cut(pd.to_numeric(df["dataset_area"], errors='coerce'), bins=q, include_lowest=True)
    return df


def plot_grouped_counts(df: pd.DataFrame, outdir: Path, top: int | None):
    needed = {"dataset_name","apriori_itemset_count","llm_itemset_count","dataset_area"}
    if not needed.issubset(df.columns):
        return
    temp = df.copy().sort_values("dataset_area", ascending=False)
    if top is None:
        top = 60 if len(temp) > 60 else len(temp)
    temp = temp.head(top)
    x = range(len(temp))
    width = 0.45
    plt.figure(figsize=(max(8, min(20, 0.3*len(temp))), 5))
    plt.bar([i - width/2 for i in x], temp["apriori_itemset_count"], width=width, label="Apriori", color="#4e79a7")
    plt.bar([i + width/2 for i in x], temp["llm_itemset_count"], width=width, label="LLM", color="#f28e2b")
    plt.xticks(list(x), temp["dataset_name"], rotation=75, ha='right', fontsize=7)
    plt.ylabel('Itemset count')
    plt.title('Apriori vs LLM itemset counts')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "grouped_counts.png", dpi=150)
    plt.close()


def plot_bin_differences(df: pd.DataFrame, outdir: Path):
    if not {"size_bin","itemset_diff"}.issubset(df.columns):
        return
    grp = df.dropna(subset=["size_bin"]).groupby("size_bin", observed=True)
    if grp.size().empty:
        return
    avg_diff = grp["itemset_diff"].mean()
    plt.figure(figsize=(8,4))
    avg_diff.plot(kind='bar', color="#59a14f")
    plt.ylabel('Average (LLM - Apriori)')
    plt.title('Average itemset count difference per size bin')
    plt.tight_layout()
    plt.savefig(outdir / "diff_bins.png", dpi=150)
    plt.close()


def plot_bin_percent_match(df: pd.DataFrame, outdir: Path):
    if not {"size_bin","percent_match"}.issubset(df.columns):
        return
    grp = df.dropna(subset=["size_bin"]).groupby("size_bin", observed=True)
    if grp.size().empty:
        return
    avg_pm = grp["percent_match"].mean()
    plt.figure(figsize=(8,4))
    (avg_pm*100).plot(kind='bar', color="#e15759")
    plt.ylabel('Percent match (%)')
    plt.title('Average percent match per size bin')
    plt.tight_layout()
    plt.savefig(outdir / "percent_match_bins.png", dpi=150)
    plt.close()


def plot_percent_match_trend(df: pd.DataFrame, outdir: Path):
    if not {"dataset_area","percent_match"}.issubset(df.columns):
        return
    temp = df.dropna(subset=["dataset_area","percent_match"]).copy()
    if temp.empty:
        return
    temp = temp.sort_values("dataset_area")
    window = max(3, int(math.sqrt(len(temp))))
    smooth = temp["percent_match"].rolling(window, min_periods=1, center=True).mean()
    plt.figure(figsize=(7,4))
    plt.scatter(temp["dataset_area"], temp["percent_match"]*100, s=16, alpha=0.6, label='Datasets')
    plt.plot(temp["dataset_area"], smooth*100, color='black', linewidth=2, label='Rolling mean')
    try:
        if (temp["dataset_area"].max() / max(1, temp["dataset_area"].min())) > 50:
            plt.xscale('log')
    except Exception:
        pass
    plt.xlabel('Dataset area (rows*cols)')
    plt.ylabel('Percent match (%)')
    plt.title('Percent match vs dataset area')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "percent_match_trend.png", dpi=150)
    plt.close()


def write_summary(df: pd.DataFrame, outdir: Path):
    cols = [c for c in [
        "dataset_name","parsed_rows","parsed_cols","dataset_area","apriori_itemset_count","llm_itemset_count",
        "itemset_diff","percent_match","size_bin","apriori_valid_ratio","llm_valid_ratio","apriori_errors","llm_errors"
    ] if c in df.columns]
    if not cols:
        return
    df[cols].to_csv(outdir / "summary_table.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Comparative Apriori vs LLM itemset analytics")
    parser.add_argument('--db', default='runs.db')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--outdir', default='visuals')
    parser.add_argument('--top', type=int, help='Top N datasets by area for grouped counts (default heuristic)')
    parser.add_argument('--bins', type=int, help='Number of quantile bins (default 5)')
    parser.add_argument('--bin-edges', nargs='*', help='Explicit numeric edges for area bins')
    parser.add_argument('--model', help='Filter to runs matching llm_model value')
    parser.add_argument('--only-passed', action='store_true', help='Restrict to validation_passed=1 runs')
    parser.add_argument('--since', help='Filter: timestamp >= since (ISO)')
    parser.add_argument('--until', help='Filter: timestamp <= until (ISO)')
    parser.add_argument('--date', help='Filter: specific date (YYYY-MM-DD) ignores since/until if set')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--tag', help='Tag suffix for output files; if omitted auto UTC timestamp')
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    try:
        df = load_runs(args.db, args.limit, args.model, args.only_passed, args.since, args.until, args.date)
    except Exception as e:
        print(f"Failed to load DB: {e}")
        return
    if df.empty:
        print("No data to visualize.")
        return

    df = parse_dimensions(df)
    df = compute_metrics(df)
    df = assign_bins(df, bins=args.bins, edges=args.bin_edges)

    # Generate with generic filenames then rename with model+tag suffix
    plot_grouped_counts(df, outdir, top=args.top)
    plot_bin_differences(df, outdir)
    plot_bin_percent_match(df, outdir)
    plot_percent_match_trend(df, outdir)
    write_summary(df, outdir)

    from datetime import datetime, UTC
    suffix_model = (args.model or 'allmodels').replace(':','_')
    suffix_tag = (args.tag or datetime.now(UTC).strftime('%Y%m%d%H%M%S'))
    suffix = f"{suffix_model}_{suffix_tag}".strip('_')
    def rename_png(base: str):
        p = outdir / f"{base}.png"
        if p.exists():
            p.rename(outdir / f"{base}_{suffix}.png")
    def rename_csv(base: str):
        p = outdir / f"{base}.csv"
        if p.exists():
            p.rename(outdir / f"{base}_{suffix}.csv")

    rename_png('grouped_counts')
    rename_png('diff_bins')
    rename_png('percent_match_bins')
    rename_png('percent_match_trend')
    rename_csv('summary_table')

    print(f"Generated comparative visuals in {outdir}")
    if args.show:
        plt.show()


def ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Extract rows, cols, and area from dataset_name pattern '<rows>x<cols>'.
    Falls back gracefully; adds columns: parsed_rows, parsed_cols, dataset_area.
    """
    if "dataset_name" not in df.columns:
        df["parsed_rows"] = pd.NA
        df["parsed_cols"] = pd.NA
        df["dataset_area"] = pd.NA
        return df
    pattern = re.compile(r"(\d+)x(\d+)(?:\.csv)?$")
    rows_list, cols_list = [], []
    for name in df["dataset_name"].astype(str):
        m = pattern.search(name)
        if m:
            rows_list.append(int(m.group(1)))
            cols_list.append(int(m.group(2)))
        else:
            rows_list.append(pd.NA)
            cols_list.append(pd.NA)
    df["parsed_rows"] = rows_list
    df["parsed_cols"] = cols_list
    # Compute area with coercion to numeric, producing NaN where missing
    pr = pd.to_numeric(df["parsed_rows"], errors='coerce')
    pc = pd.to_numeric(df["parsed_cols"], errors='coerce')
    df["dataset_area"] = pr * pc
    return df


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # difference & percent match
    if {"apriori_itemset_count","llm_itemset_count"}.issubset(df.columns):
        apr = df["apriori_itemset_count"].replace(0, pd.NA)
        df["itemset_diff"] = df["llm_itemset_count"] - df["apriori_itemset_count"]
        # Pandas 3.0+ handles inf values differently
        ratio = df["llm_itemset_count"] / apr
        ratio = ratio.replace([float('inf'), float('-inf')], pd.NA)
        # Coerce to float, invalid divisions become NaN
        df["percent_match"] = pd.to_numeric(ratio, errors='coerce')
    return df


def assign_bins(df: pd.DataFrame, bins: int | None, edges: list[float] | None) -> pd.DataFrame:
    if "dataset_area" not in df.columns:
        df["size_bin"] = pd.NA
        return df
    # Ensure numeric area series without NAType objects interfering
    try:
        area = pd.to_numeric(df["dataset_area"], errors='coerce').dropna()
    except Exception:
        area = df["dataset_area"].dropna()
    if area.empty:
        df["size_bin"] = pd.NA
        return df
    if edges:
        try:
            # ensure ascending unique
            edges = sorted(set(float(e) for e in edges))
            if len(edges) < 2:
                raise ValueError("Need at least two edges")
            df["size_bin"] = pd.cut(df["dataset_area"], bins=edges, include_lowest=True)
        except Exception as e:
            print(f"Failed to apply explicit edges ({e}); falling back to quantiles")
            edges = None
    if not edges:
        q = bins or 5
        try:
            df["size_bin"] = pd.qcut(pd.to_numeric(df["dataset_area"], errors='coerce'), q=q, duplicates='drop')
        except ValueError:
            df["size_bin"] = pd.cut(pd.to_numeric(df["dataset_area"], errors='coerce'), bins=q, include_lowest=True)
    return df


def plot_grouped_counts(df: pd.DataFrame, outdir: Path, top: int | None):
    needed = {"dataset_name","apriori_itemset_count","llm_itemset_count","dataset_area"}
    if not needed.issubset(df.columns):
        return
    temp = df.copy()
    temp = temp.sort_values("dataset_area", ascending=False)
    if top is None:
        top = 60 if len(temp) > 60 else len(temp)
    temp = temp.head(top)
    x = range(len(temp))
    width = 0.45
    plt.figure(figsize=(max(8, min(20, 0.3*len(temp))), 5))
    plt.bar([i - width/2 for i in x], temp["apriori_itemset_count"], width=width, label="Apriori", color="#4e79a7")
    plt.bar([i + width/2 for i in x], temp["llm_itemset_count"], width=width, label="LLM", color="#f28e2b")
    plt.xticks(list(x), temp["dataset_name"], rotation=75, ha='right', fontsize=7)
    plt.ylabel('Itemset count')
    plt.title('Apriori vs LLM itemset counts')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "grouped_counts.png", dpi=150)
    plt.close()


def plot_bin_differences(df: pd.DataFrame, outdir: Path):
    if not {"size_bin","itemset_diff"}.issubset(df.columns):
        return
    grp = df.dropna(subset=["size_bin"]).groupby("size_bin", observed=True)
    if grp.size().empty:
        return
    avg_diff = grp["itemset_diff"].mean()
    plt.figure(figsize=(8,4))
    avg_diff.plot(kind='bar', color="#59a14f")
    plt.ylabel('Average (LLM - Apriori)')
    plt.title('Average itemset count difference per size bin')
    plt.tight_layout()
    plt.savefig(outdir / "diff_bins.png", dpi=150)
    plt.close()


def plot_bin_percent_match(df: pd.DataFrame, outdir: Path):
    if not {"size_bin","percent_match"}.issubset(df.columns):
        return
    grp = df.dropna(subset=["size_bin"]).groupby("size_bin", observed=True)
    if grp.size().empty:
        return
    avg_pm = grp["percent_match"].mean()
    plt.figure(figsize=(8,4))
    (avg_pm*100).plot(kind='bar', color="#e15759")
    plt.ylabel('Percent match (%)')
    plt.title('Average percent match per size bin')
    plt.tight_layout()
    plt.savefig(outdir / "percent_match_bins.png", dpi=150)
    plt.close()


def plot_percent_match_trend(df: pd.DataFrame, outdir: Path):
    if not {"dataset_area","percent_match"}.issubset(df.columns):
        return
    temp = df.dropna(subset=["dataset_area","percent_match"]).copy()
    if temp.empty:
        return
    temp = temp.sort_values("dataset_area")
    # rolling window smoothing (approx LOWESS) size ~ sqrt(n)
    window = max(3, int(math.sqrt(len(temp))))
    smooth = temp["percent_match"].rolling(window, min_periods=1, center=True).mean()
    plt.figure(figsize=(7,4))
    plt.scatter(temp["dataset_area"], temp["percent_match"]*100, s=16, alpha=0.6, label='Datasets')
    plt.plot(temp["dataset_area"], smooth*100, color='black', linewidth=2, label='Rolling mean')
    plt.xscale('log') if (temp["dataset_area"].max() / max(1, temp["dataset_area"].min())) > 50 else None
    plt.xlabel('Dataset area (rows*cols)')
    plt.ylabel('Percent match (%)')
    plt.title('Percent match vs dataset area')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "percent_match_trend.png", dpi=150)
    plt.close()


def write_summary(df: pd.DataFrame, outdir: Path):
    cols = [c for c in [
        "dataset_name","parsed_rows","parsed_cols","dataset_area","apriori_itemset_count","llm_itemset_count",
        "itemset_diff","percent_match","size_bin","apriori_valid_ratio","llm_valid_ratio","apriori_errors","llm_errors"
    ] if c in df.columns]
    if not cols:
        return
    df[cols].to_csv(outdir / "summary_table.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Comparative Apriori vs LLM itemset analytics")
    parser.add_argument('--db', default='runs.db')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--outdir', default='visuals')
    parser.add_argument('--top', type=int, help='Top N datasets by area for grouped counts (default heuristic)')
    parser.add_argument('--bins', type=int, help='Number of quantile bins (default 5)')
    parser.add_argument('--bin-edges', nargs='*', help='Explicit numeric edges for area bins')
    parser.add_argument('--model', help='Filter to runs matching llm_model value')
    parser.add_argument('--only-passed', action='store_true', help='Restrict to validation_passed=1 runs')
    parser.add_argument('--since', help='Filter: timestamp >= since (ISO prefix, e.g. 2025-12-01 or 2025-12-01T10)')
    parser.add_argument('--until', help='Filter: timestamp <= until (ISO)')
    parser.add_argument('--date', help='Filter: specific date (YYYY-MM-DD) ignores since/until if set')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--tag', help='Tag suffix for output files; if omitted auto UTC timestamp')
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    try:
        df = load_runs(args.db, args.limit, args.model, args.only_passed, args.since, args.until, args.date)
    except Exception as e:
        print(f"Failed to load DB: {e}")
        return
    if df.empty:
        print("No data to visualize.")
        return

    # New comparative workflow
    df = parse_dimensions(df)
    df = compute_metrics(df)
    df = assign_bins(df, bins=args.bins, edges=args.bin_edges)

    # Dynamic filename suffix incorporating model and tag
    from datetime import datetime, UTC
    suffix_model = (args.model or 'allmodels').replace(':','_')
    suffix_tag = (args.tag or datetime.now(UTC).strftime('%Y%m%d%H%M%S'))
    suffix = f"{suffix_model}_{suffix_tag}".strip('_')

    def rename(path: Path, base: str) -> Path:
        return path / f"{base}_{suffix}.png"
    def csvname(path: Path, base: str) -> Path:
        return path / f"{base}_{suffix}.csv"

    # Temporarily redirect save paths by monkey-patching plt.savefig wrappers
    # Instead of modifying functions above heavily, we re-run logic inline with modified filenames.
    plot_grouped_counts(df, outdir, top=args.top)
    (outdir / 'grouped_counts.png').rename(rename(outdir, 'grouped_counts')) if (outdir / 'grouped_counts.png').exists() else None
    plot_bin_differences(df, outdir)
    (outdir / 'diff_bins.png').rename(rename(outdir, 'diff_bins')) if (outdir / 'diff_bins.png').exists() else None
    plot_bin_percent_match(df, outdir)
    (outdir / 'percent_match_bins.png').rename(rename(outdir, 'percent_match_bins')) if (outdir / 'percent_match_bins.png').exists() else None
    plot_percent_match_trend(df, outdir)
    (outdir / 'percent_match_trend.png').rename(rename(outdir, 'percent_match_trend')) if (outdir / 'percent_match_trend.png').exists() else None
    write_summary(df, outdir)
    (outdir / 'summary_table.csv').rename(csvname(outdir, 'summary_table')) if (outdir / 'summary_table.csv').exists() else None

    print(f"Generated comparative visuals in {outdir}")
    if args.show:
        # Re-run interactive backend for quick view if desired
        import matplotlib.pyplot as plt  # ensure same module
        plt.show()

if __name__ == '__main__':
    main()
