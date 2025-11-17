# analysis/score_calculator.py
import numpy as np

def calculate_final_score(fundamentals: dict, news_sentiment: float, macro_score: float):
    # weights
    w1, w2, w3 = 0.7, 0.2, 0.1
    
    fundamental_score = (fundamentals["E"]*0.4 + fundamentals["V"]*0.2 + 
                         fundamentals["M"]*0.15 + fundamentals["A"]*0.1 + 
                         fundamentals["C"]*0.05)
    
    score = w1 * fundamental_score + w2 * news_sentiment + w3 * macro_score
    score = np.clip(score, -1, 1)
    
    if score > 0.3:
        recommendation = "Buy"
    elif score < -0.3:
        recommendation = "Sell"
    else:
        recommendation = "Hold"
    
    return score, recommendation
