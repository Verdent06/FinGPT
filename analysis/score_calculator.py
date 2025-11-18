# analysis/score_calculator.py
import numpy as np
from analysis.llm_sentiment import LLMSentimentAnalyzer
from analysis.sentiment import mpnet_analyzer
from models import clf_handler, mpnet_embedder

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


def get_hybrid_sentiment(raw_news, mpnet_weight=0.7):
    """
    Computes combined sentiment from MPNet classifier and LLaMA model.

    Parameters:
        raw_news (list of dicts): news articles with 'title' and 'description'
        mpnet_weight (float): weight for MPNet vs LLaMA

    Returns:
        dict: {
            'mpnet_score': float,
            'llm_score': float,
            'llm_confidence': float,
            'combined_score': float,
            'combined_label': str
        }
    """

    # ----------------- MPNet sentiment -----------------
    clf = clf_handler.load_trained_clf()
    embedder = mpnet_embedder.get_embedder()
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    mpnet_results = mpnet_analyzer(raw_news, clf, embedder, label_map)
    mpnet_score = np.mean([n["sentiment_score"] for n in mpnet_results]) if mpnet_results else 0

    # ----------------- LLaMA sentiment -----------------
    llm_analyzer = LLMSentimentAnalyzer()
    llm_result = llm_analyzer.classify_sentiment(raw_news)
    llm_score = llm_result.get("sentiment_score", 0)
    llm_conf = llm_result.get("confidence", 0)

    # ----------------- Combine scores -----------------
    llm_weight = 1 - mpnet_weight
    combined_score = mpnet_weight * mpnet_score + llm_weight * llm_score * llm_conf

    # ----------------- Compute combined label -----------------
    if combined_score > 0.1:
        combined_label = "Positive"
    elif combined_score < -0.1:
        combined_label = "Negative"
    else:
        combined_label = "Neutral"

    return {
        "mpnet_score": mpnet_score,
        "llm_score": llm_score,
        "llm_confidence": llm_conf,
        "combined_score": combined_score,
        "combined_label": combined_label
    }
