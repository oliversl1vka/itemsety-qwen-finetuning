#!/usr/bin/env python3
import sqlite3
import pandas as pd
from scipy.stats import pearsonr

conn = sqlite3.connect('runs.db')
df = pd.read_sql_query('SELECT apriori_itemset_count, llm_itemset_count FROM runs', conn)
r, p = pearsonr(df['apriori_itemset_count'], df['llm_itemset_count'])

print(f'Pearson r = {r:.4f}')
print(f'p-value = {p:.2e}')
print(f'Mean Apriori: {df["apriori_itemset_count"].mean():.2f}')
print(f'Mean LLM: {df["llm_itemset_count"].mean():.2f}')
print(f'Median LLM: {df["llm_itemset_count"].median():.0f}')
print(f'Range LLM: {df["llm_itemset_count"].min():.0f}-{df["llm_itemset_count"].max():.0f}')

conn.close()
