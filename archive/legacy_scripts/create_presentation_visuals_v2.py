#!/usr/bin/env python3
"""
Generates all 6 presentation visualizations with updated data (min_support=4, 300 runs)
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

# Configuration
DB_PATH = 'runs.db'
OUTPUT_DIR = Path('visuals/presentation_final')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Professional color scheme (enhanced for visibility)
COLORS = {
    'gpt_4_1': '#1E3A8A',      # Dark blue
    'gpt_5_0': '#DC2626',       # Red
    'gpt_4o_mini': '#15803D'    # Dark green
}

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load all data from database"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
        SELECT 
            llm_model,
            dataset_name,
            dataset_size_rows,
            apriori_itemset_count,
            llm_itemset_count,
            validation_passed
        FROM runs
        WHERE validation_passed = 1
    ''', conn)
    conn.close()
    return df

def create_pipeline_architecture():
    """Slide 4: Pipeline Architecture Diagram"""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Pipeline Architektúra', 
            ha='center', va='top', fontsize=24, fontweight='bold')
    
    # Box style
    box_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                     edgecolor='black', linewidth=2)
    
    # Component positions (adjusted for proper alignment)
    components = [
        (1.5, 7, 'Dataset\n(CSV)'),
        (5, 7, 'Apriori\nAlgoritmus'),
        (8.5, 7, 'LLM\nExtrakcia'),
        (5, 4, 'Validácia\nInvariantov'),
        (5, 1, 'SQLite\nPersistencia')
    ]
    
    # Draw components
    for x, y, label in components:
        ax.text(x, y, label, ha='center', va='center', 
                fontsize=14, fontweight='bold', bbox=box_props)
    
    # Arrows with proper positioning
    arrow_props = dict(arrowstyle='->', lw=2.5, color='darkblue')
    
    # Dataset -> Apriori
    ax.annotate('', xy=(4.2, 7), xytext=(2.3, 7), arrowprops=arrow_props)
    
    # Apriori -> LLM
    ax.annotate('', xy=(7.7, 7), xytext=(5.8, 7), arrowprops=arrow_props)
    
    # LLM -> Validation (diagonal)
    ax.annotate('', xy=(5.5, 4.8), xytext=(8, 6.2), arrowprops=arrow_props)
    
    # Apriori -> Validation (diagonal)
    ax.annotate('', xy=(5, 4.8), xytext=(5, 6.2), arrowprops=arrow_props)
    
    # Validation -> Database
    ax.annotate('', xy=(5, 1.8), xytext=(5, 3.2), arrowprops=arrow_props)
    
    # Labels on arrows
    ax.text(3.2, 7.3, 'načítanie', fontsize=10, ha='center', style='italic')
    ax.text(6.7, 7.3, 'extraction', fontsize=10, ha='center', style='italic')
    ax.text(5, 3, 'persist', fontsize=10, ha='center', style='italic')
    
    # Parameters box
    params_text = 'Parametre:\n• min_support = 4\n• max_size = 3\n• temperature = 0.0'
    ax.text(9.5, 3, params_text, fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide4_pipeline_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: slide4_pipeline_architecture.png")

def create_llm_vs_apriori_scatter(df):
    """Slide 6: LLM vs Apriori Scatter Plot"""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Calculate correlation
    r, p = pearsonr(df['apriori_itemset_count'], df['llm_itemset_count'])
    
    # Scatter by model
    for model in ['gpt_4_1', 'gpt_5_0', 'gpt_4o_mini']:
        model_df = df[df['llm_model'] == model]
        ax.scatter(model_df['apriori_itemset_count'], 
                  model_df['llm_itemset_count'],
                  c=COLORS[model], label=model.replace('_', '-').upper(),
                  alpha=0.8, s=100, edgecolors='white', linewidth=0.5)
    
    # Diagonal reference line
    max_val = max(df['apriori_itemset_count'].max(), df['llm_itemset_count'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1, label='Ideálna zhoda')
    
    ax.set_xlabel('Apriori Itemset Count', fontsize=14, fontweight='bold')
    ax.set_ylabel('LLM Itemset Count', fontsize=14, fontweight='bold')
    ax.set_title(f'LLM vs Apriori Extrakcia\n(r = {r:.4f}, p < 0.001)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Legend positioning (avoid overlap)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95, 
              shadow=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide6_llm_vs_apriori.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: slide6_llm_vs_apriori.png")

def create_models_comparison(df):
    """Slide 6: Models Comparison Bar Chart"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Calculate means and std
    model_stats = df.groupby('llm_model')['llm_itemset_count'].agg(['mean', 'std']).reset_index()
    model_stats['llm_model'] = model_stats['llm_model'].str.replace('_', '-').str.upper()
    
    x = np.arange(len(model_stats))
    bars = ax.bar(x, model_stats['mean'], yerr=model_stats['std'], 
                  color=[COLORS[m.lower().replace('-', '_')] for m in model_stats['llm_model']],
                  capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Priemerný počet itemsetov', fontsize=14, fontweight='bold')
    ax.set_title('Porovnanie modelov (100 datasetov každý)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_stats['llm_model'], fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, model_stats['mean'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_val:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide6_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: slide6_models_comparison.png")

def create_histogram_distribution(df):
    """Slide 6: Distribution Histogram"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Histogram for each model
    for model in ['gpt_4_1', 'gpt_5_0', 'gpt_4o_mini']:
        model_df = df[df['llm_model'] == model]
        ax.hist(model_df['llm_itemset_count'], bins=15, 
               color=COLORS[model], alpha=0.6, 
               label=model.replace('_', '-').upper(),
               edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Počet itemsetov (LLM)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frekvencia', fontsize=14, fontweight='bold')
    ax.set_title('Distribúcia počtu itemsetov', fontsize=16, fontweight='bold', pad=20)
    
    # Legend positioning (upper right with spacing)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95,
              shadow=True, fancybox=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide6_histogram_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: slide6_histogram_distribution.png")

def create_dataset_dimensions(df):
    """Slide 6: Dataset Dimensions Box Plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Parse dimensions from dataset names
    df['rows'] = df['dataset_name'].str.extract(r'_(\d+)x')[0].astype(int)
    df['cols'] = df['dataset_name'].str.extract(r'x(\d+)\.')[0].astype(int)
    
    # Box plot for rows
    rows_data = [df[df['llm_model'] == m]['rows'].values 
                 for m in ['gpt_4_1', 'gpt_5_0', 'gpt_4o_mini']]
    bp1 = ax1.boxplot(rows_data, labels=['GPT-4.1', 'GPT-5.0', 'GPT-4o-mini'],
                      patch_artist=True, widths=0.6)
    
    for patch, model in zip(bp1['boxes'], ['gpt_4_1', 'gpt_5_0', 'gpt_4o_mini']):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Počet riadkov', fontsize=12, fontweight='bold')
    ax1.set_title('Rozmery datasetov - Riadky', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Box plot for cols
    cols_data = [df[df['llm_model'] == m]['cols'].values 
                 for m in ['gpt_4_1', 'gpt_5_0', 'gpt_4o_mini']]
    bp2 = ax2.boxplot(cols_data, labels=['GPT-4.1', 'GPT-5.0', 'GPT-4o-mini'],
                      patch_artist=True, widths=0.6)
    
    for patch, model in zip(bp2['boxes'], ['gpt_4_1', 'gpt_5_0', 'gpt_4o_mini']):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Počet stĺpcov', fontsize=12, fontweight='bold')
    ax2.set_title('Rozmery datasetov - Stĺpce', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide6_dataset_dimensions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: slide6_dataset_dimensions.png")

def create_heatmap_dimensions(df):
    """Slide 6: Heatmap of Dataset Dimensions"""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Parse dimensions
    df['rows'] = df['dataset_name'].str.extract(r'_(\d+)x')[0].astype(int)
    df['cols'] = df['dataset_name'].str.extract(r'x(\d+)\.')[0].astype(int)
    
    # Create bins
    row_bins = [0, 20, 40, 60, 80, 100]
    col_bins = [0, 25, 50, 75, 100]
    
    df['row_bin'] = pd.cut(df['rows'], bins=row_bins, labels=['0-20', '20-40', '40-60', '60-80', '80-100'])
    df['col_bin'] = pd.cut(df['cols'], bins=col_bins, labels=['0-25', '25-50', '50-75', '75-100'])
    
    # Count successful runs in each bin
    heatmap_data = df.groupby(['row_bin', 'col_bin']).size().unstack(fill_value=0)
    
    # Normalize to percentage (3 models * runs per bin)
    max_possible = df.groupby(['row_bin', 'col_bin']).size().max()
    heatmap_pct = (heatmap_data / max_possible * 100).fillna(0)
    
    sns.heatmap(heatmap_pct, annot=True, fmt='.0f', cmap='YlOrRd', 
                cbar_kws={'label': 'Úspešnosť validácie (%)'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Počet stĺpcov', fontsize=14, fontweight='bold')
    ax.set_ylabel('Počet riadkov', fontsize=14, fontweight='bold')
    ax.set_title('Pokrytie dimenzií datasetov', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide6_heatmap_dimensions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: slide6_heatmap_dimensions.png")

def main():
    print("\n" + "="*60)
    print("GENEROVANIE PREZENTAČNÝCH VIZUALIZÁCIÍ")
    print("="*60 + "\n")
    
    # Load data
    print("Načítavam dáta z databázy...")
    df = load_data()
    print(f"✓ Načítaných {len(df)} runov\n")
    
    # Generate all visualizations
    print("Generujem vizualizácie...\n")
    create_pipeline_architecture()
    create_llm_vs_apriori_scatter(df)
    create_models_comparison(df)
    create_histogram_distribution(df)
    create_dataset_dimensions(df)
    create_heatmap_dimensions(df)
    
    print("\n" + "="*60)
    print(f"✅ HOTOVO! Všetky vizualizácie uložené do: {OUTPUT_DIR}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
