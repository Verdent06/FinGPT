import yfinance as yf
from typing import Dict, Any
from analysis.fundamentals import get_fundamentals
from analysis.score_calculator import get_hybrid_sentiment, calculate_final_score, get_recommendation_label, calculate_fundamental_score
from data.news_handler import fetch_company_news
from data.yahoo_handler import get_stock_info
from utils.helpers import extract_company_name
from models import llm_handler
from data.training_manager import log_training_example

class StockAnalysisService:
    """
    Centralized service layer for orchestrating the entire FinGPT analysis pipeline.
    This class manages data flow, calls analytical models, and calculates final scores.
    """

    def __init__(self, models: Dict[str, Any] = None):
        """
        Initializes the service. Models are injected via the 'models' dictionary 
        (e.g., from FastAPI lifespan) or lazily loaded for CLI use.
        """
        self.models = models if models is not None else {}
        
        self.llm = self.models.get("llm")
        self.clf = self.models.get("clf")
        self.embedder = self.models.get("embedder")
        
        if self.llm is None:
            self.llm = llm_handler.load_llm()

    # ----------------------------------------------------------------------
    # HELPER METHODS (API Endpoint Support)
    # ----------------------------------------------------------------------

    def get_fundamentals_only(self, ticker: str) -> Dict[str, float]:
        """Fetches and returns the calculated fundamental metrics only."""
        return get_fundamentals(ticker.upper())

    def get_sentiment_result(self, ticker: str) -> Dict[str, Any]:
        """Calculates and returns the hybrid news sentiment result using injected models."""
        
        if self.clf is None or self.embedder is None:
            from models import clf_handler, mpnet_embedder
            self.clf = clf_handler.load_trained_clf()
            self.embedder = mpnet_embedder.get_embedder()
        
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
        
        sentiment_result = get_hybrid_sentiment(
            raw_news, 
            ticker, 
            clf=self.clf, 
            embedder=self.embedder, 
            llm_instance=self.llm
        )
        
        sentiment_result["num_articles"] = len(raw_news)
        sentiment_result["raw_news"] = raw_news
        return sentiment_result

    def get_score_data(self, ticker: str) -> Dict[str, Any]:
        """Calculates and returns the final score and components, excluding the LLM."""
        fundamentals_dict = self.get_fundamentals_only(ticker)
        macro_data = self.get_macro_data()
        sentiment_result = self.get_sentiment_result(ticker)
        
        final_score = calculate_final_score(
            fundamentals_dict, 
            sentiment_result["combined_score"], 
            macro_data["score"]
        )
        
        fundamental_score = calculate_fundamental_score(fundamentals_dict)
        
        recommendation = get_recommendation_label(final_score)
        
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
        """
        ticker = ticker.upper()

        info, _ = get_stock_info(ticker)
        company_name = extract_company_name(info.get("longName", ticker))
        sector = info.get("sector", "Unknown")

        fundamentals_dict = self.get_fundamentals_only(ticker)
        sentiment_data = self.get_sentiment_result(ticker)
        
        news_sentiment = sentiment_data["combined_score"]
        raw_news = sentiment_data["raw_news"]

        # FIX 3: Get final score (single float)
        final_score = calculate_final_score(
            fundamentals_dict, 
            news_sentiment, 
        )

        llm_score, llm_analysis = llm_handler.get_llm_recommendation(
            self.llm, ticker, info, fundamentals_dict, final_score, news_sentiment, company_name
        )

        # FIX 4: Get fundamental score using specific function
        f_score_val = calculate_fundamental_score(fundamentals_dict)

        # FIX 5: Remove tuple unpacking that caused the error
        log_training_example(
            ticker=ticker,
            fund_score=f_score_val,
            mpnet_score=sentiment_data["mpnet_score"],
            llm_score=sentiment_data.get("llm_score", 0), # Use safer .get()
            current_price=info.get("currentPrice", 0)
        )

        final_combined_score = 0.8 * final_score + 0.2 * llm_score

        final_recommendation = get_recommendation_label(final_combined_score)

        return {
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "info": info,
            "raw_news": raw_news,
            "hybrid_sentiment": news_sentiment,
            "news_sentiment": news_sentiment,
            
            "fundamentals": fundamentals_dict,
            "sentiment": sentiment_data,
            
            "llm_score": llm_score,

            "final_score": float(final_score),
            "final_combined_score": float(final_combined_score),
            "recommendation": final_recommendation,
            "analysis": llm_analysis,
            "raw_data": {
                "current_price": info.get("currentPrice"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE")
            }
        }