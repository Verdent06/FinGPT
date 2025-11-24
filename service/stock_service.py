import yfinance as yf
from typing import Dict, Any
from analysis.fundamentals import get_fundamentals
from analysis.macro import get_macro_info, calc_macro_score
from analysis.score_calculator import get_hybrid_sentiment, calculate_final_score
from data.news_handler import fetch_company_news
from data.yahoo_handler import get_stock_info
from utils.helpers import extract_company_name
from models import llm_handler

class StockAnalysisService:
    """
    Centralized service layer for orchestrating the entire FinGPT analysis pipeline.
    This class manages data flow, calls analytical models, and calculates final scores.
    """

    def __init__(self, models: Dict[str, Any] = None):
        """
        Initializes the service with loaded ML/LLM models, if provided (for API use).
        """
        self.models = models if models is not None else {}

    # ----------------------------------------------------------------------
    # HELPER METHODS (API Endpoint Support)
    # ----------------------------------------------------------------------

    def get_fundamentals_only(self, ticker: str) -> Dict[str, float]:
        """Fetches and returns the calculated fundamental metrics only."""
        return get_fundamentals(ticker.upper())

    def get_macro_data(self) -> Dict[str, Any]:
        """
        Fetches macroeconomic indicators and calculates the macro score.
        FIX: Corrected the variable name typo here.
        """
        indicators = get_macro_info()
        macro_score = calc_macro_score(indicators)
        indicators_clean = {k: float(v) for k, v in indicators.items()}
        
        return {
            "indicators": indicators_clean,
            "score": float(macro_score)
        }

    def get_sentiment_result(self, ticker: str) -> Dict[str, Any]:
        """Calculates and returns the hybrid news sentiment result."""
        info, _ = get_stock_info(ticker.upper())
        company_name = extract_company_name(info.get("longName", ticker))
        raw_news = fetch_company_news(ticker, company_name)
        
        if not raw_news:
             return {
                "mpnet_score": 0.0,
                "llm_score": 0.0,
                "llm_confidence": 0.0,
                "combined_score": 0.0,
                "combined_label": "Neutral",
                "num_articles": 0,
                "raw_news": []
            }
        
        sentiment_result = get_hybrid_sentiment(raw_news, ticker)
        sentiment_result["num_articles"] = len(raw_news)
        sentiment_result["raw_news"] = raw_news
        return sentiment_result

    def get_score_data(self, ticker: str) -> Dict[str, Any]:
        """Calculates and returns the final score and components, excluding the LLM."""
        fundamentals_dict = self.get_fundamentals_only(ticker)
        macro_data = self.get_macro_data()
        sentiment_result = self.get_sentiment_result(ticker)
        
        final_score, recommendation = calculate_final_score(
            fundamentals_dict, 
            sentiment_result["combined_score"], 
            macro_data["score"]
        )
        
        fundamental_score = (fundamentals_dict["E"]*0.4 + fundamentals_dict["V"]*0.2 + 
                             fundamentals_dict["M"]*0.15 + fundamentals_dict["A"]*0.1 + 
                             fundamentals_dict["C"]*0.05 + fundamentals_dict["S"]*0.1)

        return {
            "final_score": float(final_score),
            "recommendation": recommendation,
            "components": {
                "fundamentals_score": float(fundamental_score),
                "sentiment_score": float(sentiment_result["combined_score"]),
                "macro_score": float(macro_data["score"])
            }
        }
        
    # ----------------------------------------------------------------------
    # MAIN PIPELINE METHOD
    # ----------------------------------------------------------------------

    def analyze_stock(self, ticker: str) -> Dict[str, Any]:
        """
        Performs a full analysis and generates all scores/recommendations.
        Refactored to use helper methods for cleaner orchestration.
        """
        ticker = ticker.upper()

        info, _ = get_stock_info(ticker)
        company_name = extract_company_name(info.get("longName", ticker))
        sector = info.get("sector", "Unknown")

        fundamentals_dict = self.get_fundamentals_only(ticker)
        macro_data = self.get_macro_data()
        sentiment_data = self.get_sentiment_result(ticker)
        
        macro_score = macro_data["score"]
        news_sentiment = sentiment_data["combined_score"]
        raw_news = sentiment_data["raw_news"]

        final_score, base_recommendation = calculate_final_score(
            fundamentals_dict, 
            news_sentiment, 
            macro_score
        )

        llm = llm_handler.load_llm()
        llm_score, llm_recommendation, llm_justification = llm_handler.get_llm_recommendation(
            llm, ticker, info, fundamentals_dict, final_score, news_sentiment, macro_score, company_name
        )
        llm.close()

        # 5. Combine Scores
        final_combined_score = 0.8 * final_score + 0.2 * llm_score

        return {
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "info": info,
            "raw_news": raw_news,
            "hybrid_sentiment": news_sentiment,
            
            "fundamentals": fundamentals_dict,
            "macro": macro_data,
            "sentiment": sentiment_data,
            
            "final_score": float(final_score),
            "final_combined_score": float(final_combined_score),
            "recommendation": llm_recommendation,
            "raw_data": {
                "current_price": info.get("currentPrice"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE")
            }
        }