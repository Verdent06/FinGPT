# utils/format_data.py
import pandas as pd
import os

DATA_FILE = "data/training_data.csv"

def clean_and_format_csv():
    if not os.path.exists(DATA_FILE):
        print("No data file found to format.")
        return

    df = pd.read_csv(DATA_FILE)

    if df.empty:
        print("Data file is empty.")
        return

    if 'date' in df.columns and 'ticker' in df.columns:
        df = df.sort_values(by=['date', 'ticker'], ascending=[False, True])

    numeric_cols = ['fund_score', 'mpnet_score', 'llm_score', 'macro_score', 'price_at_analysis']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].round(4)

    df.to_csv(DATA_FILE, index=False)
    
    print("✅ Data formatted and saved successfully.")
    
    print("\n--- Current Training Data Snapshot ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    clean_and_format_csv()