import pandas as pd
from scipy.stats import spearmanr

def calculate_correlations(df, score_cols, target_col='mean_rating'):
    results = []
    for col in score_cols:
        # Drop NaN values for the specific pair to ensure scipy doesn't crash
        clean_df = df[[col, target_col]].dropna()
        
        # Calculate Spearman correlation
        corr, p_value = spearmanr(clean_df[col], clean_df[target_col])
        
        results.append({
            'Model Metric': col,
            'Spearman Rho': corr,
            'P-Value': p_value
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Pointing to the TSV we successfully generated at the end of Q1
    input_tsv = "data/processed/bnc_scored.tsv"
    print(f"Loading data from {input_tsv}...\n")
    df = pd.read_csv(input_tsv, sep='\t')
    
    # The metrics calculated in Question 1
    metrics_to_test = [
        'bigram_total_logprob',
        'bigram_avg_logprob',
        'bigram_SLOR',
        'trigram_total_logprob',
        'trigram_avg_logprob',
        'trigram_SLOR'
    ]
    
    # Run the correlations against the human acceptability rating
    corr_df = calculate_correlations(df, metrics_to_test, target_col='mean_rating')
    
    # Sort the results by strongest correlation (absolute value)
    corr_df['Abs_Rho'] = corr_df['Spearman Rho'].abs()
    corr_df = corr_df.sort_values(by='Abs_Rho', ascending=False).drop(columns=['Abs_Rho'])
    
    print("=== Spearman Correlation Results ===")
    print(corr_df.to_string(index=False))
    print("\n==================================")
    print("Ready for your brief note on which scores/models work best!")