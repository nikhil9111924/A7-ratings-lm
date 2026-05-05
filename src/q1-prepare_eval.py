import pandas as pd
import os

def extract_sentences_for_srilm(csv_path, output_txt_path, text_column='text'):
    """Extracts sentences from a TSV/CSV and writes them to a text file for SRILM."""
    print(f"Loading dataset from {csv_path}...")
    
    # The file is tab-separated, so we must specify sep='\t'
    df = pd.read_csv(csv_path, sep='\t')
    
    # Extract the text column and drop any empty rows
    sentences = df[text_column].dropna().tolist()
    
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            clean_sentence = str(sentence).replace('\n', ' ').strip().lower()
            f.write(f"{clean_sentence}\n")
            
    print(f"Successfully exported {len(sentences)} sentences to {output_txt_path}")

if __name__ == "__main__":
    # Pointing to the data directory based on your terminal output
    bnc_csv = "/Users/nikhilsivakumar/Desktop/College/Sem6/Comp Psycholinguistics/Assignments/A7/A7-ratings-lm/data/bnc.csv" 
    output_txt = "data/processed/bnc_sentences.txt"
    
    # Calling the function with the corrected column name
    extract_sentences_for_srilm(bnc_csv, output_txt, text_column='text')