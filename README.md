# A7-ratings-lm

## Hypothesis Testing using Language Models

**Computational Psycholinguistics - Assignment 7**

This repository contains the codebase for investigating the relationship between sentence acceptability judgments and language model probability distributions. The study evaluates traditional N-Gram models (SRILM) and modern neural models (GPT-2) against human-annotated data from the BNC dataset.

## Project Structure

```
A7-ratings-lm/
├── data/
│   ├── raw/                    # Original datasets
│   │   ├── bnc.csv             # Human acceptability ratings (TSV)
│   │   └── srilm/              # SRILM binaries and Brown corpus
│   │       ├── ngram            # SRILM scoring binary
│   │       ├── ngram-count      # SRILM training binary
│   │       └── browndata/       # Brown corpus train/dev/test splits
│   └── processed/              # Generated data and parsed sentences
│       ├── bnc_sentences.txt          # Extracted sentences for SRILM
│       ├── bnc_scored.tsv             # BNC with N-Gram scores
│       ├── bnc_scored_with_gpt2.tsv   # BNC with N-Gram + GPT-2 scores
│       └── combined_word_stats.csv    # Word frequency/length statistics
├── models/                     # Trained .lm files
│   ├── unigram.lm              # Baseline for SLOR calculation
│   ├── bigram.lm               # Kneser-Ney interpolated
│   └── trigram.lm              # Kneser-Ney interpolated
├── src/                        # Source code
│   ├── q1-prepare_eval.py
│   ├── q1-train.py
│   ├── q1-score-and-metrics.py
│   ├── q2-spearmann.py
│   ├── q3-histogram.py
│   ├── q3-analysis.py
│   ├── q4-gpt2.py
│   └── q4-visualize.py
├── results/                    # Generated plots and visualizations
│   ├── q3_word_length_histogram.png
│   ├── q3_plot4_len_vs_freq.png
│   ├── q3_plot5_log10_len_vs_log10_freq.png
│   ├── q3_plot6_log10_len_vs_ic.png
│   ├── q4_correlation_comparison_bar.png
│   └── q4_scatter_comparison.png                     
└── README.md
```

## Requirements

### Python Libraries

- `pandas` & `numpy`: Data manipulation
- `scipy`: Statistical testing (Spearman & Pearson)
- `matplotlib` & `seaborn`: Visualization
- `statsmodels`: OLS Regression analysis
- `torch` & `transformers`: GPT-2 neural language modeling

### External Tools

- **SRILM Toolkit**: The `ngram` and `ngram-count` binaries are required for Question 1.
- **Docker**: Recommended for running the Linux-based SRILM binaries on macOS (ARM64/Apple Silicon) architectures.

## Execution Guide

### Question 1 & 2: N-Gram Modeling & Correlation

These scripts handle the training of language models on the Brown corpus and scoring the BNC sentences.

1. **`q1-prepare_eval.py`**: Cleans the BNC dataset and extracts sentences into a raw text format for SRILM.
2. **`q1-train.py`**: *(Run in Linux/Docker)* Trains Unigram, Bigram, and Trigram models using Kneser-Ney interpolation (for Bigram and Trigram).
3. **`q1-score-and-metrics.py`**: *(Run in Linux/Docker)* Calculates Total Log-Prob, Avg Log-Prob, and SLOR for both Bigram and Trigram models.
4. **`q2-spearmann.py`**: Calculates the Spearman correlation between model scores and human ratings.

### Question 3: Communicative Efficiency (Piantadosi et al.)

This section evaluates if word lengths are optimized for information content rather than frequency.

1. **`q3-histogram.py`**: Unified cleaning of BNC and Brown corpora to compute word frequency/length statistics and plot the word length distribution histogram.
2. **`q3-analysis.py`**: Runs OLS regression models to compare Frequency vs. Information Content as predictors of word length, computes Pearson correlation, and generates scatter plots.

### Question 4: Extra Credit (GPT-2)

1. **`q4-gpt2.py`**: Uses a pre-trained GPT-2 Transformer to score sentences, capturing long-range dependencies missed by N-Grams. Also runs a Spearman correlation comparison across all models.
2. **`q4-visualize.py`**: Produces comparative bar charts and scatter plots (e.g., `q4_correlation_comparison_bar.png`).

## Key Findings

- **Information Content**: Confirmed as a significantly better predictor of word length ($R^2 \approx 0.0422$) compared to raw frequency ($R^2 \approx 0.0037$), supporting the findings of Piantadosi et al. (2011).
- **Neural Advantage**: GPT-2's Average Log-Probability ($\rho \approx 0.421$) outperformed the best Trigram model ($\rho \approx 0.143$) by nearly 3x, demonstrating the importance of global context in sentence acceptability.