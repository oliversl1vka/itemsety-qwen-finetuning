---
name: metrics-visualization
description: Generate charts and dashboards from pipeline metrics stored in runs.db. Use for monitoring, reporting, and analysis.
---

# Metrics Visualization

Generate charts and reports from pipeline run data.

## Overview

Visualization capabilities:
- Validation pass rate over time
- Apriori vs LLM itemset comparison
- Performance metrics (duration, counts)
- Model comparison charts
- Distribution histograms

## Quick Start

```bash
python src/utils/visualization.py --db runs.db --outdir visuals
```

## Available Charts

### 1. Validation Pass Rate
```bash
python src/utils/visualization.py --chart validation-rate --db runs.db
```
Output: `visuals/validation_pass_rate.png`

### 2. Apriori vs LLM Comparison
```bash
python src/utils/visualization.py --chart apriori-vs-llm --db runs.db
```
Output: `visuals/apriori_vs_llm_comparison.png`

### 3. Duration Distribution
```bash
python src/utils/visualization.py --chart duration-hist --db runs.db
```
Output: `visuals/duration_histogram.png`

### 4. Itemset Count Distribution
```bash
python src/utils/visualization.py --chart itemset-dist --db runs.db --bins 10
```
Output: `visuals/itemset_distribution.png`

### 5. Model Performance Comparison
```bash
python src/utils/visualization.py --chart model-comparison --db runs.db
```
Output: `visuals/model_comparison.png`

## Custom Queries

### Time Series
```python
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd

conn = sqlite3.connect("runs.db")
df = pd.read_sql_query("""
    SELECT DATE(timestamp) as date, 
           AVG(CAST(validation_passed AS FLOAT)) as pass_rate
    FROM runs 
    GROUP BY DATE(timestamp)
    ORDER BY date
""", conn)

plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['pass_rate'])
plt.title('Validation Pass Rate Over Time')
plt.ylabel('Pass Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visuals/pass_rate_timeline.png')
```

### Bar Chart
```python
df = pd.read_sql_query("""
    SELECT llm_model, 
           COUNT(*) as runs,
           AVG(CAST(validation_passed AS FLOAT)) * 100 as pass_rate
    FROM runs 
    GROUP BY llm_model
""", conn)

plt.figure(figsize=(10, 6))
plt.bar(df['llm_model'], df['pass_rate'])
plt.title('Validation Pass Rate by Model')
plt.ylabel('Pass Rate (%)')
plt.savefig('visuals/model_pass_rates.png')
```

### Scatter Plot
```python
df = pd.read_sql_query("""
    SELECT apriori_itemsets_count, llm_itemsets_count, validation_passed
    FROM runs
""", conn)

plt.figure(figsize=(10, 8))
colors = ['red' if not v else 'green' for v in df['validation_passed']]
plt.scatter(df['apriori_itemsets_count'], df['llm_itemsets_count'], c=colors, alpha=0.5)
plt.xlabel('Apriori Itemsets')
plt.ylabel('LLM Itemsets')
plt.title('Apriori vs LLM Itemset Counts')
plt.savefig('visuals/apriori_llm_scatter.png')
```

## Dashboard Components

### Summary Stats Table
```python
stats = pd.read_sql_query("""
    SELECT 
        COUNT(*) as total_runs,
        SUM(validation_passed) as passed_runs,
        ROUND(AVG(CAST(validation_passed AS FLOAT)) * 100, 1) as pass_rate,
        ROUND(AVG(apriori_duration_s), 2) as avg_apriori_time,
        ROUND(AVG(llm_duration_s), 2) as avg_llm_time,
        ROUND(AVG(apriori_itemsets_count), 1) as avg_apriori_itemsets,
        ROUND(AVG(llm_itemsets_count), 1) as avg_llm_itemsets
    FROM runs
""", conn)
print(stats.to_markdown())
```

### Metrics Grid
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Chart 1: Pass Rate Pie
axes[0, 0].pie([passed, failed], labels=['Passed', 'Failed'], autopct='%1.1f%%')
axes[0, 0].set_title('Validation Status')

# Chart 2: Duration Box Plot
df.boxplot(column=['apriori_duration_s', 'llm_duration_s'], ax=axes[0, 1])
axes[0, 1].set_title('Processing Duration')

# Chart 3: Itemsets Histogram
axes[1, 0].hist(df['apriori_itemsets_count'], bins=20, alpha=0.7, label='Apriori')
axes[1, 0].hist(df['llm_itemsets_count'], bins=20, alpha=0.7, label='LLM')
axes[1, 0].legend()
axes[1, 0].set_title('Itemset Count Distribution')

# Chart 4: Timeline
df_time.plot(x='date', y='pass_rate', ax=axes[1, 1])
axes[1, 1].set_title('Pass Rate Over Time')

plt.tight_layout()
plt.savefig('visuals/dashboard.png', dpi=150)
```

## Output Directory

```
visuals/
├── validation_pass_rate.png
├── apriori_vs_llm_comparison.png
├── duration_histogram.png
├── itemset_distribution.png
├── model_comparison.png
├── dashboard.png
└── metrics_summary.md
```

## Markdown Report Generation

```python
report = f"""
# Pipeline Metrics Report

**Generated:** {datetime.now().isoformat()}

## Summary
- Total Runs: {total_runs}
- Validation Pass Rate: {pass_rate:.1f}%
- Average Apriori Duration: {avg_apriori:.2f}s
- Average LLM Duration: {avg_llm:.2f}s

## Charts
![Validation Rate](validation_pass_rate.png)
![Comparison](apriori_vs_llm_comparison.png)

## Recommendations
- Pass rate below 90%: Review LLM prompt
- LLM duration > 60s: Reduce chunk size
"""

with open("visuals/metrics_summary.md", "w") as f:
    f.write(report)
```

## Scheduling

### Daily Report
```bash
# Add to crontab
0 9 * * * cd /path/to/repo && python src/utils/visualization.py --db runs.db --outdir visuals --full
```

### After Pipeline Run
```bash
python pipeline.py --data-dir data/datasets_v2 --llm-full && \
python src/utils/visualization.py --db runs.db --outdir visuals
```

## Troubleshooting

### No data in charts
- Check runs.db has records
- Verify date range filter
- Check column names match schema

### Charts not rendering
- Install matplotlib: `pip install matplotlib`
- For headless: `matplotlib.use('Agg')`
- Check write permissions on output dir
