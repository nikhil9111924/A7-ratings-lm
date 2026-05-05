import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings

# Suppress HuggingFace warnings for cleaner output
warnings.filterwarnings('ignore')

def load_gpt2():
    print("Loading pre-trained GPT-2 model and tokenizer (this might take a minute on first run)...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval() # Set to evaluation mode
    return model, tokenizer

def score_sentence_gpt2(sentence, model, tokenizer):
    """Calculates the log10 probability of a sentence using GPT-2."""
    # Tokenize the input
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    # Handle edge case of empty or extremely short sentences
    if input_ids.size(1) < 2:
        return -99.0, 1
        
    with torch.no_grad():
        # Passing labels=input_ids automatically calculates the cross-entropy loss 
        # (which is the negative average log-likelihood)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
    # Total natural log probability
    total_ln_prob = -loss.item() * input_ids.size(1)
    
    # Convert natural log (ln) to base 10 log to match SRILM's format exactly
    total_log10_prob = total_ln_prob * np.log10(np.e)
    
    # Get word count (simple whitespace split to match the n-gram approach)
    word_count = len(sentence.split())
    
    return total_log10_prob, max(1, word_count)

if __name__ == "__main__":
    # 1. Load Data
    input_tsv = "data/processed/bnc_scored.tsv"
    print(f"Loading scored dataset from {input_tsv}...")
    df = pd.read_csv(input_tsv, sep='\t')
    
    # 2. Load Model
    model, tokenizer = load_gpt2()
    
    # 3. Score Sentences
    print(f"Scoring {len(df)} sentences with GPT-2. This may take a few minutes...")
    gpt2_total_logprobs = []
    gpt2_word_counts = []
    
    for i, sentence in enumerate(df['text'].tolist()):
        if i > 0 and i % 100 == 0:
            print(f"  ...processed {i}/{len(df)} sentences")
            
        logprob, words = score_sentence_gpt2(str(sentence), model, tokenizer)
        gpt2_total_logprobs.append(logprob)
        gpt2_word_counts.append(words)
        
    # 4. Calculate GPT-2 Metrics
    gpt2_total_logprobs = np.array(gpt2_total_logprobs)
    gpt2_word_counts = np.array(gpt2_word_counts)
    
    df['gpt2_total_logprob'] = gpt2_total_logprobs
    df['gpt2_avg_logprob'] = gpt2_total_logprobs / gpt2_word_counts
    
    # For SLOR, we use the unigram scores we already calculated in Q1
    # SLOR = Model_LogProb - Unigram_LogProb
    # First, reconstruct the unigram scores from the Q1 data: Unigram = Bigram_Total - Bigram_SLOR
    uni_logprobs = df['bigram_total_logprob'] - df['bigram_SLOR']
    df['gpt2_SLOR'] = df['gpt2_total_logprob'] - uni_logprobs
    
    # Save the updated dataframe
    output_tsv = "data/processed/bnc_scored_with_gpt2.tsv"
    df.to_csv(output_tsv, index=False, sep='\t')
    print(f"\nSaved full dataset with GPT-2 scores to {output_tsv}")
    
    # 5. Run Spearman Correlation Comparison
    print("\n=== FINAL SPEARMAN CORRELATION SHOWDOWN ===")
    metrics_to_test = [
        'bigram_total_logprob', 'trigram_total_logprob', 'gpt2_total_logprob',
        'bigram_avg_logprob', 'trigram_avg_logprob', 'gpt2_avg_logprob',
        'bigram_SLOR', 'trigram_SLOR', 'gpt2_SLOR'
    ]
    
    results = []
    for col in metrics_to_test:
        clean_df = df[[col, 'mean_rating']].dropna()
        corr, p_val = spearmanr(clean_df[col], clean_df['mean_rating'])
        results.append({'Model Metric': col, 'Spearman Rho': corr})
        
    corr_df = pd.DataFrame(results)
    corr_df['Abs_Rho'] = corr_df['Spearman Rho'].abs()
    corr_df = corr_df.sort_values(by='Abs_Rho', ascending=False).drop(columns=['Abs_Rho'])
    
    print(corr_df.to_string(index=False))
    print("===========================================")