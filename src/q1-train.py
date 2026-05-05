import subprocess
import os

def train_srilm_model(order, input_file, output_model, use_kn=True):
    """Trains an n-gram model using SRILM."""
    print(f"Training {order}-gram model...")
    command = [
        "data/raw/srilm/ngram-count",
        "-text", input_file,
        "-order", str(order),
        "-lm", output_model
    ]
    
    # Apply Kneser-Ney interpolation for Bigram and Trigram
    if use_kn and order > 1:
        command.extend(["-kndiscount", "-interpolate"])
        
    try:
        subprocess.run(command, check=True)
        print(f"Successfully saved model to {output_model}")
    except subprocess.CalledProcessError as e:
        print(f"Error training {order}-gram: {e}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    brown_train_path = "data/raw/srilm/browndata/brown-train.txt"
    
    # Train Unigram (Baseline for SLOR), Bigram, and Trigram
    train_srilm_model(1, brown_train_path, "models/unigram.lm", use_kn=False)
    train_srilm_model(2, brown_train_path, "models/bigram.lm", use_kn=True)
    train_srilm_model(3, brown_train_path, "models/trigram.lm", use_kn=True)