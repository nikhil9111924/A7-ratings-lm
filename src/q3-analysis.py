import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm
import os

if __name__ == "__main__":
    print("Loading data...")
    # Load the statistics generated in the previous step
    df = pd.read_csv("data/processed/combined_word_stats.csv")
    
    # --- Data Preparation ---
    total_tokens = df['frequency'].sum()
    
    # 1. Calculate Probability of each word
    df['probability'] = df['frequency'] / total_tokens
    
    # 2. Calculate Information Content: -log2(Probability)
    df['information_content'] = -np.log2(df['probability'])
    
    # 3. Calculate log10 values for the plots
    df['log10_length'] = np.log10(df['length'])
    df['log10_frequency'] = np.log10(df['frequency'])
    
    # --- Task 1: Regression Models ---
    print("\n=== 1. Regression Analysis ===")
    # Model A: Predict Length from Frequency
    X_freq = sm.add_constant(df['frequency'])
    model_freq = sm.OLS(df['length'], X_freq).fit()
    
    # Model B: Predict Length from Information Content
    X_ic = sm.add_constant(df['information_content'])
    model_ic = sm.OLS(df['length'], X_ic).fit()
    
    print(f"Model A (Frequency predicts Length) R-squared: {model_freq.rsquared:.4f}")
    print(f"Model B (Info Content predicts Length) R-squared: {model_ic.rsquared:.4f}")
    if model_ic.rsquared > model_freq.rsquared:
        print("-> Result: Information Content has a better fit to the data.")
    else:
        print("-> Result: Frequency has a better fit to the data.")

    # --- Task 2: Shortest Words ---
    print("\n=== 2. Shortest Words ===")
    shortest_len = df['length'].min()
    shortest_words = df[df['length'] == shortest_len]['word'].tolist()
    print(f"The shortest words in the dataset have a length of {shortest_len} letter(s).")
    # Print the first 20 as an example, ignoring any weird blank space artifacts
    print(f"Examples: {[w for w in shortest_words if w.strip()][:20]}") 

    # --- Task 3: Pearson Correlation ---
    print("\n=== 3. Pearson Correlation ===")
    pearson_corr, p_val = pearsonr(df['length'], df['frequency'])
    print(f"Pearson coefficient between length and frequency: {pearson_corr:.4f}")

    # --- Tasks 4, 5, 6: Plots ---
    print("\n=== Generating Plots ===")
    os.makedirs("results", exist_ok=True)
    
    # Set a consistent style
    plot_kwargs = {'alpha': 0.3, 'color': 'royalblue', 'edgecolors': 'none', 's': 20}

    # Plot 4: Length vs Frequency
    plt.figure(figsize=(8, 5))
    plt.scatter(df['length'], df['frequency'], **plot_kwargs)
    plt.title("4. Word Length vs Frequency")
    plt.xlabel("Word Length (Number of Letters)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("results/q3_plot4_len_vs_freq.png")
    plt.close()

    # Plot 5: log10(Length) vs log10(Frequency)
    plt.figure(figsize=(8, 5))
    plt.scatter(df['log10_length'], df['log10_frequency'], **plot_kwargs)
    plt.title("5. Log10(Word Length) vs Log10(Frequency)")
    plt.xlabel("Log10(Word Length)")
    plt.ylabel("Log10(Frequency)")
    plt.tight_layout()
    plt.savefig("results/q3_plot5_log10_len_vs_log10_freq.png")
    plt.close()

    # Plot 6: log10(Length) vs Information Content
    plt.figure(figsize=(8, 5))
    plt.scatter(df['log10_length'], df['information_content'], color='darkorange', alpha=0.3, s=20)
    plt.title("6. Log10(Word Length) vs Information Content")
    plt.xlabel("Log10(Word Length)")
    plt.ylabel("Information Content (-log2(P))")
    plt.tight_layout()
    plt.savefig("results/q3_plot6_log10_len_vs_ic.png")
    plt.close()

    print("SUCCESS: All models calculated and plots saved to the 'results/' directory.")