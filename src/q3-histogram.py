import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from collections import Counter

def load_and_clean_text():
    """Loads BNC and Brown datasets, lowers cases, and extracts clean words."""
    print("Extracting words from BNC dataset...")
    # Load BNC
    bnc_df = pd.read_csv("data/raw/bnc.csv", sep='\t')
    bnc_text = " ".join(bnc_df['text'].dropna().astype(str).tolist())
    
    print("Extracting words from Brown corpus...")
    # Load Brown (train, dev, test)
    brown_dir = "data/raw/srilm/browndata"
    brown_text = ""
    if os.path.exists(brown_dir):
        for filename in os.listdir(brown_dir):
            filepath = os.path.join(brown_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    brown_text += " " + f.read()
    else:
        print(f"Warning: Could not find Brown data at {brown_dir}")

    # Combine all text
    combined_text = bnc_text + " " + brown_text
    
    # Lowercase and extract only alphabetical words (strips punctuation and numbers)
    print("Cleaning text and removing punctuation...")
    words = re.findall(r'\b[a-z]+\b', combined_text.lower())
    return words

if __name__ == "__main__":
    # 1. Get all words
    all_words = load_and_clean_text()
    print(f"Total words extracted: {len(all_words):,}")
    
    # 2. Calculate frequencies and lengths
    print("Calculating frequencies and lengths...")
    word_counts = Counter(all_words)
    
    # Create a DataFrame for easy manipulation in the next steps
    df_words = pd.DataFrame.from_dict(word_counts, orient='index', columns=['frequency']).reset_index()
    df_words = df_words.rename(columns={'index': 'word'})
    df_words['length'] = df_words['word'].apply(len)
    
    # Save this processed data so we don't have to clean the text again for the Q3 regressions
    os.makedirs("data/processed", exist_ok=True)
    df_words.to_csv("data/processed/combined_word_stats.csv", index=False)
    print("Saved word statistics to data/processed/combined_word_stats.csv")
    
    # 3. Plot the Histogram of Word Lengths (Weighted by how many times the word appears)
    print("Generating histogram...")
    
    # To get an accurate representation of word lengths in the corpus, we repeat the length by its frequency
    # (e.g., if "the" has length 3 and appears 100 times, we add 100 3s to the plot)
    lengths_expanded = np.repeat(df_words['length'].values, df_words['frequency'].values)
    
    plt.figure(figsize=(10, 6))
    plt.hist(lengths_expanded, bins=range(1, 21), edgecolor='black', alpha=0.7, align='left')
    plt.title('Distribution of Word Lengths in Combined Dataset (BNC + Brown)')
    plt.xlabel('Word Length (Number of Letters)')
    plt.ylabel('Frequency (Total Occurrences)')
    plt.xticks(range(1, 21))
    plt.grid(axis='y', alpha=0.75)
    
    # Save the plot for your report
    os.makedirs("results", exist_ok=True)
    plot_path = "results/q3_word_length_histogram.png"
    plt.savefig(plot_path)
    print(f"SUCCESS: Histogram saved to {plot_path}")