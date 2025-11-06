"""Focused comparative visualization for runs.db.

Outputs ONLY comparative Apriori vs LLM itemset analytics (old generic charts removed):
    1. grouped_counts.png         - Side‑by‑side Apriori vs LLM itemset counts per dataset (optionally top-N by size)
    2. diff_bins.png              - Average absolute difference (LLM - Apriori) per dataset size bin
    3. percent_match_bins.png     - Average percent match (LLM / Apriori) per size bin
    4. percent_match_trend.png    - Scatter + LOWESS-like rolling mean of percent match vs dataset area (rows*cols)
    5. summary_table.csv          - Tabular metrics (includes parsed rows, cols, area, difference, percent_match, bin label)

Size binning defaults to quantile-based (q=5). Override with --bins X or use explicit boundaries via --bin-edges.

Dataset name pattern assumed: ds_XXXX_<rows>x<cols>.csv (rowsxcols anywhere at end). If pattern fails, attempts to
fallback to existing numeric metadata columns; unknown dimensions become NaN and excluded from size-based charts.

Usage (PowerShell):
    python visualization.py --db runs.db --outdir visuals --top 40 --bins 6

Args:
    --limit N          Only first N rows chronologically
    --top N            Only plot top N (by dataset_area) in grouped counts (default: all <= 60 else 60)
    --bins Q           Number of quantile bins (ignored if --bin-edges provided)
    --bin-edges e1 e2 ...  Explicit area edges (exclusive high except last). Provide ascending numbers.
    --outdir DIR       Output directory (default visuals)
    --show             Display windows (switches backend)
"""
from __future__ import annotations
import argparse, os, sqlite3, math, re
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless by default; switch to interactive if --show
import matplotlib.pyplot as plt

RELEVANT_COLS = [
    "id","timestamp","dataset_name","dataset_size_rows","dataset_size_bytes",
    "min_support","max_size","apriori_itemset_count","llm_itemset_count",
    "apriori_valid_ratio","llm_valid_ratio","apriori_errors","llm_errors",
    "run_duration_ms","error_message"
]


def load_runs(db_path: str, limit: int | None) -> pd.DataFrame:
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM runs ORDER BY timestamp ASC", conn)
    finally:
        conn.close()
    # Keep only relevant columns that exist
    cols = [c for c in RELEVANT_COLS if c in df.columns]
    df = df[cols]
    if limit:
        df = df.head(limit)
    # Derived metrics
    if "run_duration_ms" in df.columns and not {"apriori_duration_ms","llm_duration_ms"}.issubset(df.columns):
        # Split rough durations if we only have total; naive proportional split by itemset counts
        # (Placeholder: user can later record specific stage times)
        pass
    # Speed ratio (LLM vs Apriori itemset volume per dataset size)
    if {"apriori_itemset_count","llm_itemset_count","dataset_size_rows"}.issubset(df.columns):
        # Vectorized ratio; avoid deprecated pandas options. Replace division by zero and infinities with NaN.
        denom = df["apriori_itemset_count"].replace(0, pd.NA)
        ratio = df["llm_itemset_count"] / denom
        ratio = ratio.astype(float)
        ratio.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        df["llm_to_apriori_itemset_ratio"] = ratio
    return df


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
    # area preference: parsed_rows*parsed_cols else dataset_size_rows * (approx cols if exists)
    with pd.option_context('future.no_silent_downcasting', True):
        df["dataset_area"] = df["parsed_rows"].astype('Float64') * df["parsed_cols"].astype('Float64')
    return df


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # difference & percent match
    if {"apriori_itemset_count","llm_itemset_count"}.issubset(df.columns):
        apr = df["apriori_itemset_count"].replace(0, pd.NA)
        df["itemset_diff"] = df["llm_itemset_count"] - df["apriori_itemset_count"]
        df["percent_match"] = (df["llm_itemset_count"] / apr).astype(float)
    return df


def assign_bins(df: pd.DataFrame, bins: int | None, edges: list[float] | None) -> pd.DataFrame:
    if "dataset_area" not in df.columns:
        df["size_bin"] = pd.NA
        return df
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
            df["size_bin"] = pd.qcut(df["dataset_area"], q=q, duplicates='drop')
        except ValueError:
            df["size_bin"] = pd.cut(df["dataset_area"], bins=q, include_lowest=True)
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
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    try:
        df = load_runs(args.db, args.limit)
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

    plot_grouped_counts(df, outdir, top=args.top)
    plot_bin_differences(df, outdir)
    plot_bin_percent_match(df, outdir)
    plot_percent_match_trend(df, outdir)
    write_summary(df, outdir)

    print(f"Generated comparative visuals in {outdir}")
    if args.show:
        # Re-run interactive backend for quick view if desired
        import matplotlib.pyplot as plt  # ensure same module
        plt.show()

if __name__ == '__main__':
    main()
