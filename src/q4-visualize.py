import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os

def calculate_correlations(df, metrics):
    """Calculates Spearman correlation for a list of metrics against mean_rating."""
    results = []
    for col in metrics:
        clean_df = df[[col, 'mean_rating']].dropna()
        corr, _ = spearmanr(clean_df[col], clean_df['mean_rating'])
        
        # Determine model type and metric type for easier plotting
        if 'gpt2' in col:
            model = 'GPT-2'
        elif 'trigram' in col:
            model = 'Trigram'
        else:
            model = 'Bigram'
            
        if 'total' in col:
            m_type = 'Total Log-Prob'
        elif 'avg' in col:
            m_type = 'Avg Log-Prob'
        else:
            m_type = 'SLOR'
            
        results.append({
            'Metric_Name': col,
            'Model': model,
            'Metric_Type': m_type,
            'Spearman_Rho': corr,
            'Absolute_Rho': abs(corr)
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Loading final dataset...")
    df = pd.read_csv("data/processed/bnc_scored_with_gpt2.tsv", sep='\t')
    
    metrics = [
        'bigram_total_logprob', 'trigram_total_logprob', 'gpt2_total_logprob',
        'bigram_avg_logprob', 'trigram_avg_logprob', 'gpt2_avg_logprob',
        'bigram_SLOR', 'trigram_SLOR', 'gpt2_SLOR'
    ]
    
    corr_df = calculate_correlations(df, metrics)
    os.makedirs("results", exist_ok=True)
    
    print("Generating Comparative Bar Chart...")
    # --- Plot 1: Bar Chart of Correlations ---
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Create a grouped bar chart
    g = sns.barplot(
        data=corr_df, 
        x='Metric_Type', 
        y='Spearman_Rho', 
        hue='Model',
        palette=['#4C72B0', '#55A868', '#C44E52'] # Distinct colors
    )
    
    plt.title('Spearman Correlation with Human Acceptability Ratings', fontsize=14, pad=15)
    plt.xlabel('Scoring Metric', fontsize=12)
    plt.ylabel('Spearman Correlation Coefficient (ρ)', fontsize=12)
    plt.legend(title='Model Type')
    plt.tight_layout()
    plt.savefig("results/q4_correlation_comparison_bar.png", dpi=300)
    plt.close()
    
    print("Generating Scatter Plot Comparison...")
    # --- Plot 2: Scatter Plot (Best N-Gram vs Best GPT-2) ---
    # We will plot Trigram Total Log-Prob vs GPT-2 Avg Log-Prob to see the clustering difference
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    plot_kwargs = {'alpha': 0.4, 's': 15, 'edgecolors': 'none'}
    
    # Subplot A: Trigram
    axes[0].scatter(df['trigram_total_logprob'], df['mean_rating'], color='#55A868', **plot_kwargs)
    axes[0].set_title('A. Trigram (Total Log-Prob)', fontsize=12)
    axes[0].set_xlabel('Log Probability', fontsize=11)
    axes[0].set_ylabel('Mean Human Rating', fontsize=11)
    
    # Subplot B: GPT-2
    axes[1].scatter(df['gpt2_avg_logprob'], df['mean_rating'], color='#C44E52', **plot_kwargs)
    axes[1].set_title('B. GPT-2 (Avg Log-Prob)', fontsize=12)
    axes[1].set_xlabel('Log Probability', fontsize=11)
    
    plt.suptitle('Predicting Human Acceptability: N-Gram vs. Neural Model', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("results/q4_scatter_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("SUCCESS: Graphs saved to the 'results/' directory.")