# analysis/train_weights.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

DATA_FILE = "data/training_data.csv"
CONFIG_FILE = "config/config.json"

def train_and_update_weights():
    print("--- Starting Quant Analysis ---")
    
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print("❌ No training data found.")
        return
    
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows of trading data.")
    
    # 2. Prepare Features (X) and Target (y)
    # We ignore Macro for now because it's constant (0.5) in our backfill
    feature_cols = ['fund_score', 'mpnet_score', 'llm_score']
    X = df[feature_cols]
    y = df['target_return']
    
    # 3. Train the Model
    # We force positive coefficients (weights) because negative weights 
    # would mean "Good news = Sell", which is illogical for this system.
    model = LinearRegression(positive=True) 
    model.fit(X, y)
    
    # 4. Extract Weights
    raw_weights = model.coef_
    intercept = model.intercept_
    
    w_fund = raw_weights[0]
    w_mpnet = raw_weights[1]
    w_llm = raw_weights[2]
    
    print("\n--- Regression Results ---")
    print(f"Base Return (Intercept): {intercept:.4f}")
    print(f"Weight: Fundamentals:    {w_fund:.4f}")
    print(f"Weight: MPNet Sentiment: {w_mpnet:.4f}")
    print(f"Weight: LLaMA Sentiment: {w_llm:.4f}")
    
    # 5. Normalize to sum to 1.0 (for Config)
    # We want weights that fit into: Score = w_f*Fund + w_s*Sent + w_m*Macro
    # Since Macro was constant, we'll keep its config weight fixed (e.g. 0.1) 
    # and distribute the rest (0.9) based on what we learned.
    
    total_learned_weight = w_fund + w_mpnet + w_llm
    
    if total_learned_weight == 0:
        print("❌ Model could not find a signal. Using defaults.")
        return

    # Let's say we reserve 0.1 for Macro (fixed)
    available_weight = 0.9 
    
    # Normalize learned weights
    norm_fund = (w_fund / total_learned_weight) * available_weight
    norm_total_sent = ((w_mpnet + w_llm) / total_learned_weight) * available_weight
    
    # Calculate the internal split for sentiment (MPNet vs LLaMA)
    if (w_mpnet + w_llm) > 0:
        mpnet_split = w_mpnet / (w_mpnet + w_llm)
    else:
        mpnet_split = 0.5 # Default if both are 0
        
    print(f"\n--- Suggested Config Configuration ---")
    print(f"Fundamentals: {norm_fund:.2f}")
    print(f"Sentiment:    {norm_total_sent:.2f}")
    print(f"Macro:        0.10 (Fixed)")
    print(f"MPNet Split:  {mpnet_split:.2f}")
    print(f"LLaMA Split:  {1-mpnet_split:.2f}")

    # 6. Update Config (Optional: Auto-save)
    save = input("\nDo you want to update config.json with these values? (y/n): ")
    if save.lower() == 'y':
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        
        # Update Weights
        config['weights']['fundamentals'] = round(norm_fund, 2)
        config['weights']['sentiment'] = round(norm_total_sent, 2)
        config['weights']['macro'] = 0.1
        
        # We need to save the MPNet/LLaMA split. 
        # Currently config.json doesn't store this (it's hardcoded 0.7 in code).
        # Let's add it to config.
        config['sentiment_split'] = {
            "mpnet": round(mpnet_split, 2),
            "llm": round(1-mpnet_split, 2)
        }
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print("✅ config.json updated.")

if __name__ == "__main__":
    train_and_update_weights()