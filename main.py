# main.py
import warnings
import logging
from utils import logging_setup, helpers
from service.stock_service import StockAnalysisService
import sys

# ------------ Warnings Suppressing ---------------
warnings.simplefilter("ignore")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


# ----------------- Setup Logging -----------------
logger = logging_setup.setup_logging()

def main():
    try:
        # ----------------- User Input -----------------
        ticker = input("\nEnter a stock ticker symbol (e.g., AAPL): ").upper()

        if not ticker:
                print("No ticker provided. Exiting.")
                return

        print(f"Initializing analysis for {ticker}...")
        
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

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred during analysis: {e}")
        print(f"An error occurred. Check logs for details: {e}")

if __name__ == "__main__":
    main()