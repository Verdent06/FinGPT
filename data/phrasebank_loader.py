# data/phrasebank_loader.py
import pandas as pd

def load_phrasebank(file_path="data/Sentences_75Agree.txt"):
    """
    Load labeled sentences for financial news classifier.
    Returns a pandas DataFrame with columns ['sentence', 'label'].
    """
    data = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or '@' not in line:
            continue
        sentence, label = line.rsplit('@', 1)
        data.append((sentence.strip(), label.strip()))
    
    df = pd.DataFrame(data, columns=['sentence', 'label'])
    return df
