# analysis/score_calculator.py
import numpy as np
import os
import json
from datetime import datetime
from analysis.mpnet_sentiment import mpnet_analyzer
from analysis.llm_sentiment import LLMSentimentAnalyzer
from data.sentiment_cache import get_cached_sentiment, update_cache 
from utils.config_loader import CONFIG  # Use the robust loader we already have

MASTER_LOG_FILE = "logs/sentiment_master.json"

def get_recommendation_label(score: float) -> str:
    """
    Centralized logic for converting a numerical score (-1 to 1) 
    into a text label using thresholds from config.
    """
    # Load thresholds from config (safely)
    thresholds = CONFIG.get("thresholds", {})
    buy_th = thresholds.get("buy", 0.3)
    sell_th = thresholds.get("sell", -0.3)

    if score > buy_th:
        return "Buy"
    elif score < sell_th:
        return "Sell"
    else:
        return "Hold"

def calculate_fundamental_score(fundamentals: dict) -> float:
    """
    Calculates the weighted fundamental score.
    Separated for cleaner architecture and logging.
    """
    # FIX: 'fund_weights' is at the top level of config.json
    fw = CONFIG.get("fund_weights", {})
    
    score = (
        fundamentals.get("E", 0) * fw.get("E", 0.4) +
        fundamentals.get("V", 0) * fw.get("V", 0.2) +
        fundamentals.get("M", 0) * fw.get("M", 0.15) +
        fundamentals.get("A", 0) * fw.get("A", 0.1) +
        fundamentals.get("C", 0) * fw.get("C", 0.05) +
        fundamentals.get("S", 0) * fw.get("S", 0.1)
    )
    return score

def calculate_final_score(fundamentals: dict, news_sentiment: float, macro_score: float) -> float:
    """
    Calculates the final multi-factor score.
    Returns a SINGLE float (fixing the multiplication error).
    """
    # Load top-level weights
    weights = CONFIG.get("weights", {})
    
    # 1. Get Fundamental Score
    fund_score = calculate_fundamental_score(fundamentals)
    
    # 2. Calculate Final Weighted Score
    # Keys match config.json: "fundamentals", "sentiment", "macro"
    final_score = (
        weights.get("fundamentals", 0.7) * fund_score + 
        weights.get("sentiment", 0.2) * news_sentiment + 
        weights.get("macro", 0.1) * macro_score
    )
    
    return np.clip(final_score, -1, 1)

def get_hybrid_sentiment(raw_news, ticker, clf, embedder, llm_instance, mpnet_weight=0.7):  
    # 1. MPNet Sentiment
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    mpnet_results = mpnet_analyzer(raw_news, clf, embedder, label_map)
    mpnet_score = np.mean([n["sentiment_score"] for n in mpnet_results]) if mpnet_results else 0

    # 2. LLaMA Sentiment (With Caching)
    llm_analyzer = LLMSentimentAnalyzer(llm_instance)
    llm_scores = []
    
    print(f"Processing {len(raw_news)} articles for LLaMA sentiment...")
    
    for article in raw_news:
        text_key = f"{article.get('title','')}. {article.get('description','')}"
        
        # Check Cache
        cached_data = get_cached_sentiment(text_key)
        
        if cached_data:
            score = cached_data['score']
            conf = cached_data['confidence']
        else:
            print(f"  [LLaMA Running] {article.get('title')[:30]}...")
            llm_result = llm_analyzer.analyze_single_article(article)
            
            update_cache(text_key, llm_result)
            
            score = llm_result['sentiment_score']
            conf = llm_result.get('confidence', 1.0)
            
        llm_scores.append(score * conf)
    
    final_llm_score = np.mean(llm_scores) if llm_scores else 0
    
    # 3. Combine Scores
    llm_weight = 1 - mpnet_weight
    combined_score = mpnet_weight * mpnet_score + llm_weight * final_llm_score
    
    # 4. Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_entry = {
        "timestamp": timestamp,
        "ticker": ticker,
        "summary": {
            "mpnet_score": mpnet_score,
            "llm_score": final_llm_score,
            "combined_score": combined_score,
            "combined_label": get_recommendation_label(combined_score) # Re-use label logic or keep simple
        },
        "articles": mpnet_results
    }
    
    # Safe Append to Master Log
    if os.path.exists(MASTER_LOG_FILE):
        try:
            with open(MASTER_LOG_FILE, "r") as f:
                master_data = json.load(f)
                if not isinstance(master_data, list): master_data = []
        except:
            master_data = []
    else:
        master_data = []
        
    master_data.append(log_entry)
    with open(MASTER_LOG_FILE, "w") as f:
        json.dump(master_data, f, indent=4)
    
    return {
        "mpnet_score": mpnet_score,
        "llm_score": final_llm_score,
        "combined_score": combined_score,
        "combined_label": "Positive" if combined_score > 0.1 else "Negative" if combined_score < -0.1 else "Neutral"
    }