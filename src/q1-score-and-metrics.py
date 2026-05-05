import subprocess
import pandas as pd
import numpy as np
import re

def score_sentences_srilm(model_path, text_file):
    """Runs SRILM ngram to evaluate sentences and parses the log probabilities."""
    print(f"Scoring sentences with {model_path}...")
    
    command = [
        "data/raw/srilm/ngram",
        "-lm", model_path,
        "-ppl", text_file,
        "-debug", "1"
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    logprobs = []
    word_counts = []
    
    # Track the words for the current sentence across lines
    current_words = None
    
    for line in result.stdout.split('\n'):
        # 1. Catch the word count line (e.g., "1 sentences, 10 words, 0 OOVs")
        words_match = re.search(r'(\d+)\s+words?', line)
        if words_match and "sentences" in line:
            current_words = max(1, int(words_match.group(1)))
            
        # 2. Catch the logprob line (e.g., "0 zeroprobs, logprob= -25.31 ppl= 120.1")
        if "logprob=" in line:
            logprob_match = re.search(r'logprob=\s*([-\d.]+)', line)
            
            # If we found a logprob AND we remembered the word count from the line above
            if logprob_match and current_words is not None:
                logprobs.append(float(logprob_match.group(1)))
                word_counts.append(current_words)
                current_words = None # Reset for the next sentence
                
    # The last matched pair in SRILM output is the file summary. We exclude it.
    return np.array(logprobs[:-1]), np.array(word_counts[:-1])

if __name__ == "__main__":
    # Corrected path to match your actual directory structure
    bnc_csv = "data/raw/bnc.csv"
    sentences_txt = "data/processed/bnc_sentences.txt"
    
    # 1. Get raw log probabilities and word counts
    uni_logprobs, _ = score_sentences_srilm("models/unigram.lm", sentences_txt)
    bi_logprobs, word_counts = score_sentences_srilm("models/bigram.lm", sentences_txt)
    tri_logprobs, _ = score_sentences_srilm("models/trigram.lm", sentences_txt)
    
    # 2. Load the dataset (Tab Separated)
    df = pd.read_csv(bnc_csv, sep='\t')
    
    # 3. Drop empty text rows to match our sentence extraction
    df = df.dropna(subset=['text']).copy()
    
    # Safety Check: Truncate to the shortest array to prevent length crashes
    min_len = min(len(uni_logprobs), len(df))
    if len(uni_logprobs) != len(df):
        print(f"Warning: SRILM scored {len(uni_logprobs)} sentences, but dataset has {len(df)} rows. Syncing...")
        
    uni_logprobs = uni_logprobs[:min_len]
    bi_logprobs = bi_logprobs[:min_len]
    tri_logprobs = tri_logprobs[:min_len]
    word_counts = word_counts[:min_len]
    df = df.iloc[:min_len].copy()
    
    # 4. Compute Bigram Metrics
    df['bigram_total_logprob'] = bi_logprobs
    df['bigram_avg_logprob'] = bi_logprobs / word_counts
    df['bigram_SLOR'] = bi_logprobs - uni_logprobs
    
    # 5. Compute Trigram Metrics
    df['trigram_total_logprob'] = tri_logprobs
    df['trigram_avg_logprob'] = tri_logprobs / word_counts
    df['trigram_SLOR'] = tri_logprobs - uni_logprobs
    
    # Save the final dataset
    output_csv = "data/processed/bnc_scored.tsv"
    df.to_csv(output_csv, index=False, sep='\t')
    
    print(f"Metrics successfully computed and saved to {output_csv}. Ready for Question 2!")