# main.py
import warnings
import logging
from utils import logging_setup, helpers
from service.stock_service import StockAnalysisService

# ------------ Warnings Suppressing ---------------
warnings.simplefilter("ignore")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


# ----------------- Setup Logging -----------------
logger = logging_setup.setup_logging()

# ----------------- Main Analysis Logic (Refactored) -----------------

try:
    # ----------------- User Input -----------------
    ticker = input("\nEnter a stock ticker symbol (e.g., AAPL): ").upper()

    analysis_service = StockAnalysisService() 

    results = analysis_service.analyze_stock(ticker)

    # ----------------- Display Results -----------------
    helpers.display_results(
        ticker, 
        results["info"], 
        results["raw_news"], 
        results["hybrid_sentiment"], 
        results["final_score"], 
        results["final_combined_score"], 
        results["recommendation"]
    )

except Exception as e:
    logger.error(f"An unexpected error occurred during analysis: {e}")