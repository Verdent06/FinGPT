# data/training_manager.py
import os
import pandas as pd
from datetime import datetime

DATA_FILE = "data/training_data.csv"

def log_training_example(ticker, fund_score, mpnet_score, llm_score, macro_score, current_price):
    """
    Saves a single analysis snapshot to a CSV file.
    This creates the dataset for future regression training.
    """
    
    # 1. Define the row structure
    new_row = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "ticker": ticker,
        "fund_score": fund_score,
        "mpnet_score": mpnet_score,
        "llm_score": llm_score,
        "macro_score": macro_score,
        "price_at_analysis": current_price,
        "target_return": None # To be filled later by a separate script
    }
    
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.DataFrame(columns=new_row.keys())
        
    mask = (df['date'] == new_row['date']) & (df['ticker'] == new_row['ticker'])
    if not df[mask].empty:
        df.loc[mask, ["fund_score", "mpnet_score", "llm_score", "macro_score", "price_at_analysis"]] = \
            [fund_score, mpnet_score, llm_score, macro_score, current_price]
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
    df.to_csv(DATA_FILE, index=False)